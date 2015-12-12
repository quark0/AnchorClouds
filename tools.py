from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import cm 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import time

def visualize_datapoints(X, y, title = ""):
    d = X.shape[1]
    assert d == 2 or d == 3, "only 2/3-D datapoints can be visualized"

    fig = pyplot.figure()
    if d == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2], c = y)
    if d == 2:
        ax = fig.add_subplot(111)
        ax.scatter(X[:,0],X[:,1], c = y)

    ax.set_title(title)
    fig.show()

def visualize_anchors(X, A):
    '''visualize the anchors'''
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Anchors")

    ax.scatter(X[:,0], X[:,1], X[:,2], alpha=0.1)
    ax.scatter(A[:,0], A[:,1], A[:,2], s=60, c='r', marker='^')

    fig.show()

def visualize_edges(X, A, Z, threshold, title = ""):
    '''Visualize the unweighted instance-anchor edges'''
    d = X.shape[1]
    assert d == 2 or d == 3, "only 2/3-D edges can be visualized"

    links = np.where(Z>threshold)
    # source and target vertices
    s = X[links[0],:]
    t = A[links[1],:]

    fig = pyplot.figure()
    color=cm.rainbow(np.linspace(0, 1, A.shape[0]))

    if d == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(10,-75)
        edge = lambda i:([s[i,0], t[i,0]], [s[i,1], t[i,1]], [s[i,2], t[i,2]])
    if d == 2:
        ax = fig.add_subplot(111)
        edge = lambda i:([s[i,0], t[i,0]], [s[i,1], t[i,1]])

    for i in xrange(s.shape[0]):
        ax.plot(*edge(i), c=color[links[1][i],:], alpha=0.6)

    ax.set_title(title)
    fig.show()

def visualize_eigenvectors(U, k):
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    for j in xrange(k):
        ax.plot(U[:,j])
    fig.show()

def random_data_split(n, n_labeled, n_trials):
    ls, us = [], []
    for trial in xrange(n_trials):
        l = np.random.choice(n, n_labeled, replace=False)
        ls.append(l)
        u = np.setdiff1d(np.arange(n), l)
        us.append(u)
    return ls, us

def print_formated_results(r):
    if len(r) > 0:
        print "\nAccuracy:"
        for alg in r.keys():
            print '%10s' % alg,
        print

        n_trials = len(r.values()[0])
        for trial in xrange(n_trials):
            for alg in r.keys():
                print '%10.2f' % r[alg][trial],
            print

        print "\nMean Accuracy:"
        for alg in r.keys():
            print '%10s' % alg,
        print
        for alg in r.keys():
            print '%10.2f' % np.mean(r[alg]),
        print

def kmeans_centroids(X, n_clusters):
    t_start = time.time()

    km = MiniBatchKMeans(n_clusters=n_clusters,\
            init='k-means++', max_iter=5, init_size=2*n_clusters, batch_size=500).fit(X)
    A = km.cluster_centers_

    # A = split_by_spatial_tree(X, n_clusters)

    t_elapsed = time.time() - t_start
    print 'kmeans: %.3f secs' % t_elapsed

    return A

def split_by_spatial_tree(X, n_anchors):
    '''Data partitioning via spatial trees
    Dependency: http://cseweb.ucsd.edu/~naverma/SpatialTrees/index.html

    Args:
        X: matrix of data points
   
    Returns:
        A: centroids for each spatial partitioning
    '''
    from spatialtree import spatialtree

    height = np.log2(n_anchors)
    height_int = np.int(height)
    if height_int != height:
        print "number of anchors is not power of 2"

    T = spatialtree(X, rule='rp', height=height_int, spill=0.0, min_items=1)

    A = np.zeros((n_anchors, X.shape[1]))

    c = 0
    for t in T.traverse():
        if t.isLeaf():
            indices = [index for index in t.__iter__()]
            A[c,:] = np.average(X[indices, :], axis=0)
            c = c + 1

    return A

