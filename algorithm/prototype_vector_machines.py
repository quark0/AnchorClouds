import tools
import numpy as np 
import manifold_generator
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import rbf_kernel
import time

class PVM:
    '''
    Prototype Vector Machines for large-scale semi-supervised learning

    http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6803073
    '''

    def __init__(self, n_prototypes, sigma2):
        self.n_prototypes = n_prototypes
        self.sigma2 = sigma2

    def fit(self, X):
        A = tools.kmeans_centroids(X, self.n_prototypes)
        self.W = rbf_kernel(A, A, gamma = 1./self.sigma2)
        self.H = rbf_kernel(X, A, gamma = 1./self.sigma2)

        self.W_dagger = np.linalg.inv(self.W)

        d_tilde = self.H.dot(self.W_dagger.dot(self.H.T.sum(axis=1)))
        self.HtH = self.H.T.dot(self.H)
        self.HtSH = (self.H.T * d_tilde).dot(self.H) - self.HtH.dot(self.W_dagger).dot(self.HtH.T)
        self.n = X.shape[0]

    def predict(self, l, Yl, C1, C2):
        u = np.setdiff1d(np.arange(self.n), l)

        Hl = self.H[l,:]

        # NOTE A trick to avoid the expensive Hu.T.dot(Hu)
        M = self.HtSH + (C1-C2)*Hl.T.dot(Hl) + C2*self.HtH

        # M = self.HtSH + C1*Hl.T.dot(Hl) + C2*Hu.T.dot(Hu)

        fv = np.linalg.pinv(M).dot(Hl.T).dot(Yl)
        scores = self.H.dot(fv)

        return scores

if __name__ == '__main__':
    np.random.seed(1267)
    X, Y, y = manifold_generator.double_swiss_roll(n_samples=1000, var=.8)
    n_trials = 5
    n_labeled = 24

    n = X.shape[0]
    ls, us = tools.random_data_split(n, n_labeled, n_trials)

    n_clusters = 48
    sigma2 = 1
    C1 = 1
    C2 = 1

    pvm = PVM(n_clusters, sigma2)
    pvm.fit(X)

    for trial in xrange(n_trials):
        l, u = ls[trial], us[trial]
        # scores = prototype_vector_machines(X, l, Y[l,:])
        scores = pvm.predict(l, Y[l,:], C1, C2)

        y_hat = np.argmax(normalize(scores, axis=0, norm='l1'), axis=1)
        acc = 100.*np.sum(y_hat[u] == y[u])/len(u)

        print acc

