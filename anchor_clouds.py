import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import tools
import time

class Anchor:
    inner_dim = -1
    def __init__(self, a, X):
        '''
        Args:
            a: anchor centroid
            X: data belonging to the anchor
        '''
        m, p = X.shape
        U = X - a   # local coordinate for X
        Phi, s, _ = svds( U.T/np.sqrt(m), k=self.inner_dim )
        s2 = s**2

        # initialize ppca parameters
        self.sigma2 = ( (U*U).sum()/m - s2.sum() ) / (p - self.inner_dim)
        self.a = a
        #self.W = Phi.dot( np.diag( np.sqrt(s2 - self.sigma2) ) )
        self.W = Phi * np.sqrt(s2 - self.sigma2) 
        self.M = np.linalg.inv(self.W.T.dot(self.W) + self.sigma2*np.eye(self.inner_dim))
        self.det_sqrt = np.prod( np.sqrt(s2 + self.sigma2) )

    def log_ppca_density(self, x):
        u = x - self.a
        Wtu = self.W.T.dot(u)
        return -0.5*(sum(u*u) - Wtu.T.dot(self.M).dot(Wtu))/self.sigma2 - np.log(self.det_sqrt)

def anchor_clouds(X, inner_dim, n_anchors, n_data_per_anchor, n_anchor_per_data):
    # find anchors centroids
    A, labels = tools.kmeans_centroids(X, n_anchors)

    # find nearest datapoints for anchors
    #start = time.time()
    #_, nbrs_of_A = NearestNeighbors(n_neighbors = n_data_per_anchor).fit(X).kneighbors(A)
    #print 'Nearest data search: %.3f secs' % (time.time() - start)

    # initialize anchors (including estimating local ppca models)
    start = time.time()

    Anchor.inner_dim = inner_dim
    anchors = []
    for j in xrange(n_anchors):
        #anchors.append(Anchor(A[j,:], X[nbrs_of_A[j,:],:]))
        nbr_samples = np.random.choice(np.where(labels==j)[0], n_data_per_anchor)
        anchors.append(Anchor(A[j,:], X[nbr_samples,:]))

    print 'Constructing clouds: %.3f secs' % (time.time() - start)

    #for i in xrange(n):
        #for j in xrange(n_anchors):
            #Z[i,j] = anchors[j].ppca_density(X[i,:])
    #nbrs_of_X = Z.argsort()[:,-3:]

    # find nearest anchors for datapoints
    start = time.time()
    _, nbrs_of_X = NearestNeighbors(n_neighbors = n_anchor_per_data).fit(A).kneighbors(X)
    print 'Nearest anchor search: %.3f secs' % (time.time() - start)

    # compute probability assignment with the "exp-normalize" trick
    n = X.shape[0]
    Z = np.zeros((n, n_anchors))

    start = time.time()
    for i in xrange(n):

        max_log_density = -1e10
        for j in nbrs_of_X[i,:]:
            Z[i,j] = anchors[j].log_ppca_density(X[i,:])
            if Z[i,j] > max_log_density:
                max_log_density = Z[i,j]

        density_sum = 0
        for j in nbrs_of_X[i,:]:
            Z[i,j] = np.exp(Z[i,j] - max_log_density)
            density_sum = density_sum + Z[i,j]

        for j in nbrs_of_X[i,:]:
            Z[i,j] = Z[i,j] / density_sum

    print 'Constructing Z: %.3f secs' % (time.time() - start)

    return A, Z

if __name__ == '__main__':
    np.random.seed(1267)

    import manifold_generator
    n_anchors = 32
    n_data_per_anchor = 20
    n_anchor_per_data = 3
    inner_dim = 1
    X, _, _ = manifold_generator.swiss_roll(n_samples=300, var=.75)
    A, Z = anchor_clouds(X, 1, n_anchors, n_data_per_anchor, n_anchor_per_data)

    tools.visualize_edges(X, A, Z, 1e-12)
