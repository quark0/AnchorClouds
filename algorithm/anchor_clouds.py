import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import time

import manifold_generator
import tools

class AnchorClouds:
    def __init__(self, inner_dim, n_anchors, n_data_per_anchor, n_anchor_per_data):
        self.inner_dim = inner_dim
        self.n_anchors = n_anchors
        self.n_data_per_anchor = n_data_per_anchor
        self.n_anchor_per_data = n_anchor_per_data

    def fit(self, X):
        # find anchors centroids
        A = tools.kmeans_centroids(X, self.n_anchors)

        # find nearest datapoints for anchors
        start = time.time()
        _, nbrs_of_A = NearestNeighbors(n_neighbors = self.n_data_per_anchor).fit(X).kneighbors(A)
        print 'Nearest data search: %.3f secs' % (time.time() - start)

        # initialize anchors (including estimating local ppca models)
        start = time.time()

        anchors = []
        for j in xrange(self.n_anchors):
            anchors.append(Anchor(A[j,:], X[nbrs_of_A[j,:],:], self.inner_dim))
            # print np.where(labels==j)[0]
            # nbr_samples = np.random.choice(np.where(labels==j)[0], self.n_data_per_anchor)
            # anchors.append(Anchor(A[j,:], X[nbr_samples,:]))

        print 'Constructing clouds: %.3f secs' % (time.time() - start)

        #for i in xrange(n):
            #for j in xrange(self.n_anchors):
                #Z[i,j] = anchors[j].ppca_density(X[i,:])
        #nbrs_of_X = Z.argsort()[:,-3:]

        # find nearest anchors for datapoints
        start = time.time()
        _, nbrs_of_X = NearestNeighbors(n_neighbors = self.n_anchor_per_data).fit(A).kneighbors(X)
        print 'Nearest anchor search: %.3f secs' % (time.time() - start)

        # compute probability assignment with the "exp-normalize" trick
        n = X.shape[0]
        self.Z = np.zeros((n, self.n_anchors))

        start = time.time()
        for i in xrange(n):

            max_log_density = -1e10
            for j in nbrs_of_X[i,:]:
                self.Z[i,j] = anchors[j].log_ppca_density(X[i,:])
                if self.Z[i,j] > max_log_density:
                    max_log_density = self.Z[i,j]

            density_sum = 0
            for j in nbrs_of_X[i,:]:
                self.Z[i,j] = np.exp(self.Z[i,j] - max_log_density)
                density_sum = density_sum + self.Z[i,j]

            for j in nbrs_of_X[i,:]:
                self.Z[i,j] = self.Z[i,j] / density_sum

        print 'Constructing Z: %.3f secs' % (time.time() - start)

    def predict(self, l, Yl, gamma):
         scores = tools.reduced_sml(self.Z, l, Yl, gamma)
         return scores

class Anchor:
    def __init__(self, a, X, inner_dim):
        '''
        Args:
            a: anchor centroid
            X: data belonging to the anchor
        '''
        self.inner_dim = inner_dim
        m, p = X.shape
        U = X - a   # local coordinate for X
        Phi, s, _ = svds( U.T/np.sqrt(m), k=self.inner_dim )
        s2 = s**2

        # initialize ppca parameters
        self.sigma2 = ( np.linalg.norm(U, 'fro')**2/m - s2.sum() ) / (p - self.inner_dim)
        self.a = a

        self.W = Phi * np.sqrt(s2 - self.sigma2) 
        self.det_sqrt = np.prod(np.sqrt(s2 + self.sigma2))

        try:
            self.M = np.linalg.inv(self.W.T.dot(self.W) + self.sigma2*np.eye(self.inner_dim))
        except np.linalg.LinAlgError:
            print "singular anchor"

    def log_ppca_density(self, x):
        u = x - self.a
        Wtu = self.W.T.dot(u)
        return -0.5*(u.dot(u) - Wtu.T.dot(self.M).dot(Wtu))/self.sigma2 - np.log(self.det_sqrt)

if __name__ == '__main__':
    np.random.seed(1267)

    n_anchors = 32
    n_data_per_anchor = 20
    n_anchor_per_data = 3
    inner_dim = 1
    X, _, _ = manifold_generator.double_swiss_roll(n_samples=300, var=.75)
    A, Z = anchor_clouds(X, 1, n_anchors, n_data_per_anchor, n_anchor_per_data)

    tools.visualize_edges(X, A, Z, 1e-12)
