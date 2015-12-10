import numpy as np
import cPickle, gzip
from sklearn.manifold import TSNE

def convert_to_Y(y):
    n = y.shape[0]
    Y = np.zeros((n, np.unique(y).size))
    Y[np.arange(n), y] = 1
    return Y

def double_swiss_roll(n_samples = 250, var = 1.):
    '''details about the generating procedure can be found at
    http://people.cs.uchicago.edu/~dinoj/manifold/swissroll.html'''

    # generate gaussian clusters
    means = 2.5*np.array([[+1,+1], [-1,-1], [+1,-1], [-1,+1]], dtype=float) + 10
    A = np.vstack(tuple([np.random.multivariate_normal(mean, var*np.eye(2), n_samples)\
            for mean in means]))
    y = np.vstack(tuple([i*np.ones((n_samples,1),dtype=int) for i in xrange(means.shape[0])]))[:,0]

    # roll up the gaussian clusters
    a1, a2 = A[:,0], A[:,1]
    X = np.column_stack((a1*np.cos(a1), a2, a1*np.sin(a1)))

    Y = convert_to_Y(y)

    return X, Y, y

def mnist(fid = 'data/mnist.pkl.gz'):
    f = gzip.open(fid, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    X, y = train_set
    Y = convert_to_Y(y)

    #model = TSNE(n_components=2, random_state=0)
    #X = model.fit_transform(X)

    return X, Y, y

def usps(fid = 'data/zip.train'):
   Xy = np.loadtxt(fid) 

   y = np.int32(Xy[:,0])
   X = Xy[:,1:]
   Y = convert_to_Y(y)
   
   return X, Y, y

