import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import nnls
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from sklearn import mixture
from sklearn.preprocessing import normalize
import time
import manifold_generator
import tools

def locally_anchor_embedding(X, A, idx):
    '''Locally Anchor Embedding

    Args:
        X: matrix of data points
        A: matrix of anchors
        idx: mapping from each element in X to anchor indices

    Rreturns: 
        Z: transition probability from X to A
    '''
    # Warning: no regularization is imposed over Z
    # Warning: this is a simplified heuristic solution different from the original paper
    n = X.shape[0]
    beta = np.vstack([nnls(A[idx[i,:],:].T, X[i,:])[0] for i in xrange(n)])
    Z = normalize(beta, axis=1, norm='l1')

    return Z

def reduced_sml(Z, l, Yl, gamma):
    '''dimension-reduced semi-supervised learning

    Please refer to the anchor graph paper for more details.
    '''
    Lambda_inv = np.diag(1./Z.sum(axis=0))

    # sparse operations are crucial for large Z
    Z_sparse = csr_matrix(Z)
    ZZ_sparse = Z_sparse.T.dot(Z_sparse)
    ZZ = ZZ_sparse.toarray()

    L_tilde = ZZ - ZZ.dot(Lambda_inv).dot(ZZ)
    Zl = Z[l,:]
    A = np.linalg.lstsq(Zl.T.dot(Zl) + gamma*L_tilde, Zl.T)[0].dot(Yl)
    return Z.dot(A)

def reduced_sml_eigen(U, Sigma, l, Yl, gamma):
    '''partial eigendecomposition for graph Laplacian approximation

    Refs: "Semi-supervised learning in gigantic image collections." NIPS'09
    '''
    Ul = U[l,:]
    A = np.linalg.lstsq(gamma*Sigma + Ul.T.dot(Ul), Ul.T)[0].dot(Yl)
    return U.dot(A)

def Nystrom():
    '''Nystrom's method for graph Laplacian approximation'''
    pass

def anchor_points(X, n_clusters, n_nbrs):
    '''AnchorGraph construction

    Args:
        A: anchors via clustering
        Z: anchor embedings from X to A, where each row sums up to 1
    '''
    n = X.shape[0]

    A, _ = tools.kmeans_centroids(X, n_clusters)

    nbrs = NearestNeighbors(n_neighbors = n_nbrs, metric='euclidean').fit(A)
    nbrs_distances, nbrs_idx = nbrs.kneighbors(X)

    nbrs_Z = locally_anchor_embedding(X, A, nbrs_idx)

    Z = np.zeros((n, n_clusters))
    Z[np.arange(n)[:,np.newaxis], nbrs_idx] = nbrs_Z

    return A, Z

# FIXME: for high dimesion, in gmm many diagonal components of ZZ become 0
def anchor_points_gmm(X, n_clusters, n_nbrs, kmeans_center=True):
    '''AnchorClound construction via Exact Gaussian Mixture'''

    gmm = mixture.GMM(n_components=n_clusters, covariance_type='full', min_covar=1e-2)

    if kmeans_center == True:
        # fix the GMM means to be the kmeans centers
        gmm.params = 'wc'
        gmm.init_params = 'wc'
        gmm.means_, _ = tools.kmeans_centroids(X, n_clusters)

    gmm.fit(X)

    A = gmm.means_
    Z = gmm.predict_proba(X)

    return A, Z

def laplacian_eigen(X, n_clusters, n_nbrs):
    '''Obtain the top-k eigensystem of the graph Laplacian

    The eigen solver adopts shift-invert mode as described in
    http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
    '''
    from scipy.sparse import csgraph
    nbrs = NearestNeighbors(n_neighbors=n_nbrs).fit(X)

    # NOTE W is a dense graph thus may lead to memory leak
    W = nbrs.kneighbors_graph(X).toarray()

    W_sym = np.maximum(W, W.T)
    L = csr_matrix(csgraph.laplacian(W_sym, normed=True))
    [Sigma, U] = eigsh(L, n_clusters+1, sigma=0, which='LM')

    # remove the trivial (smallest) eigenvalues & vectors
    return U[:,1:], Sigma[1:]

if __name__ == '__main__':
    from anchor_clouds import anchor_clouds

    np.random.seed(1267)
    dataset     = 'mnist'
    n_trials    = 10
    visualize   = False

    # data generation
    if dataset == 'mnist':
        X, Y, y = manifold_generator.mnist()
        n_nbrs      = 3
        n_clusters  = 128
        n_labeled   = 100
        inner_dim   = 5
        gamma       = 1e-4
        algs = ["anchor_points", "anchor_clouds"]
    elif dataset == 'swiss':
        X, Y, y = manifold_generator.swiss_roll(n_samples=2000, var=.75)
        n_nbrs      = 3
        n_clusters  = 32
        n_labeled   = 48
        inner_dim   = 1
        gamma       = 1e-3
        # heuristics: keep m = O(n/d)
        algs = ["anchor_points", "anchor_clouds", "anchor_points_gmm"]
    else:
        sys.exit('invalid dataset')
    n = X.shape[0]
    n_data_per_anchor = max(n/n_clusters/inner_dim, 100)
    #tools.visualize_datapoints(X, y, "Ground Truth")

    ls, us = tools.random_data_split(n, n_labeled, n_trials)

    results = {}
    for alg in algs:

        print alg

        t_start = time.time()
        if alg == "anchor_points":
            A, Z = anchor_points(X, n_clusters, n_nbrs)
        if alg == "anchor_points_gmm":
            A, Z = anchor_points_gmm(X, n_clusters, n_nbrs, False)
        if alg == "anchor_clouds":
            A, Z = anchor_clouds(X, inner_dim, n_clusters, n_data_per_anchor, n_nbrs)
        if alg == "exact_eigen":
            U, Sigma = laplacian_eigen(X, n_clusters, n_nbrs)
            #tools.visualize_eigenvectors(U, 3)

        if visualize == True and alg.startswith("anchor"):
            tools.visualize_edges(X, A, Z, 1e-6, alg)
            
        t_elapsed = time.time() - t_start
        print '%.3f secs' % t_elapsed

        results[alg] = []
        for trial in xrange(n_trials):

            print "%d" % trial

            l, u = ls[trial], us[trial]

            if alg.startswith("anchor"):
                scores = reduced_sml(Z, l, Y[l,:], gamma)
            if alg == "exact_eigen":
                scores = reduced_sml_eigen(U, Sigma, l, Y[l,:], gamma)

            y_hat = np.argmax(normalize(scores, axis=0, norm='l1'), axis=1)

            acc = 100.*np.sum(y_hat[u] == y[u])/len(u)
            results[alg].append(acc)

    tools.print_formated_results(results)

