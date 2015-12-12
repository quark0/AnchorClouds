import tools
import numpy as np
from sklearn import mixture
from sklearn.neighbors import NearestNeighbors

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

        nbrs_Z = tools.locally_anchor_embedding(X, A, nbrs_idx)

        self.Z = np.zeros((self.n, self.n_clusters))
        self.Z[np.arange(self.n)[:,np.newaxis], nbrs_idx] = nbrs_Z

    def predict(self, l, Yl, gamma):
        scores = tools.reduced_sml(self.Z, l, Yl, gamma)
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

