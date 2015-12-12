import numpy as np
import tools
from scipy.sparse import csgraph
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

class LaplacianEigen:
    def __init__(self, n_clusters, n_nbrs):
        self.n_clusters = n_clusters
        self.n_nbrs = n_nbrs

    def fit(self, X):
        '''Obtain the top-k eigensystem of the graph Laplacian

        The eigen solver adopts shift-invert mode as described in
        http://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
        '''
        nbrs = NearestNeighbors(n_neighbors=self.n_nbrs).fit(X)

        # NOTE W is a dense graph thus may lead to memory leak
        W = nbrs.kneighbors_graph(X).toarray()
        W_sym = np.maximum(W, W.T)

        L = csr_matrix(csgraph.laplacian(W_sym, normed=True))
        [Sigma, U] = eigsh(L, self.n_clusters+1, sigma=0, which='LM')

        # remove the trivial (smallest) eigenvalues & vectors
        self.Sigma, self.U = Sigma[1:], U[:,1:]

        #tools.visualize_eigenvectors(U, 3)
    
    def predict(self, l, Yl, gamma):
        ''' Refs: "Semi-supervised learning in gigantic image collections." NIPS'09 '''
        Ul = self.U[l,:]
        A = np.linalg.lstsq(gamma*self.Sigma + Ul.T.dot(Ul), Ul.T)[0].dot(Yl)
        scores = self.U.dot(A)
        return scores
