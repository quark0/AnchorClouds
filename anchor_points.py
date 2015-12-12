import tools
import numpy as np
from scipy.sparse import csr_matrix
from sklearn import mixture
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import nnls
from sklearn.preprocessing import normalize

def locally_anchor_embedding(X, A, idx):
    '''Locally Anchor Embedding

    Args:
        X: matrix of data points
        A: matrix of anchors
        idx: mapping from each element in X to anchor indices

    Returns: 
        Z: transition probability from X to A
    '''
    # Warning: no regularization is imposed over Z
    # Warning: this is a simplified heuristic solution different from the original paper
    beta = np.vstack([nnls(A[idx[i,:],:].T, X[i,:])[0] for i in xrange(X.shape[0])])
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

class AnchorPoints:

    def __init__(self, n_clusters, n_nbrs):
        self.n_clusters = n_clusters
        self.n_nbrs = n_nbrs

    def fit(self, X):
        '''AnchorGraph construction

        Variables:
            A: ancors via clustering
            Z: anchor embedings from X to A, where each row sums up to 1
        '''
        self.n = X.shape[0]

        A = tools.kmeans_centroids(X, self.n_clusters)

        nbrs = NearestNeighbors(n_neighbors = self.n_nbrs, metric='euclidean').fit(A)
        nbrs_distances, nbrs_idx = nbrs.kneighbors(X)

        nbrs_Z = locally_anchor_embedding(X, A, nbrs_idx)

        self.Z = np.zeros((self.n, self.n_clusters))
        self.Z[np.arange(self.n)[:,np.newaxis], nbrs_idx] = nbrs_Z

    def predict(self, l, Yl, gamma):
        scores = reduced_sml(self.Z, l, Yl, gamma)
        return scores

class AnchorPointsGMM(AnchorPoints):

    def fit(self, X, kmeans_center=False):
        '''AnchorClound construction via Exact Gaussian Mixture'''

        gmm = mixture.GMM(n_components=self.n_clusters, covariance_type='full', min_covar=1e-2)

        if kmeans_center == True:
            # fix the GMM means to be the kmeans centers
            gmm.params = 'wc'
            gmm.init_params = 'wc'
            gmm.means_ = tools.kmeans_centroids(X, self.n_clusters)

        gmm.fit(X)

        self.Z = gmm.predict_proba(X)

