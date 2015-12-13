import numpy as np
import sys
from sklearn.preprocessing import normalize
import time
import manifold_generator

from algorithm import tools
from algorithm import nearest_neighbors
from algorithm import anchor_points
from algorithm import anchor_clouds
from algorithm import prototype_vector_machines
from algorithm import laplacian_eigen
from algorithm import anchor_points

if __name__ == '__main__':

    np.random.seed(1267)
    dataset     = 'letter'
    n_trials    = 20

    C1, C2 =  1, 1

    #MNIST
    if dataset == 'mnist':
        n_nbrs      = 3
        n_clusters  = 512
        n_labeled   = 100
        inner_dim   = 6
        gamma       = 1e-1
        n_data_per_anchor = 300
        algs = ["ap", "ac", "nn", "pvm"]
        sigma2 = 10

    #Letter.scale
    elif dataset == 'letter':
        n_nbrs      = 3
        n_clusters  = 512
        n_labeled   = 256
        inner_dim   = 4
        gamma       = 1e-2
        n_data_per_anchor = 100
        algs = ["ac"]
        sigma2 = 1

    #USPS
    elif dataset == 'usps':
        n_nbrs      = 2
        n_clusters  = 200
        n_labeled   = 100
        inner_dim   = 5
        gamma       = 1e-1
        n_data_per_anchor = 40
        algs = ["ap", "ac", "nn"]
        sigma2 = 50

    #Double Swiss Roll
    elif dataset == 'swiss':
        n_nbrs      = 3
        n_clusters  = 48
        n_labeled   = 24
        inner_dim   = 1
        gamma       = 1e-2
        n_data_per_anchor = 200
        # heuristics: keep m = O(n/d)
        algs = ["ap", "ac", "nn", "pvm", "apg"]
        sigma2 = 1

    data = {
            "swiss"  : manifold_generator.double_swiss_roll(n_samples=1000,  var=.8),
            "usps"   : manifold_generator.usps(),
            "letter" : manifold_generator.letter(),
            "mnist"  : manifold_generator.mnist()
            }

    model = {
            "ap" : (
                anchor_points.AnchorPoints(n_clusters, n_nbrs),
                (gamma, )),
            "ac" : (
                anchor_clouds.AnchorClouds(inner_dim, n_clusters, n_data_per_anchor, n_nbrs),
                (gamma, )),
            "nn" : (
                nearest_neighbors.NearestNeighbors(k=1),
                ()),
            "le" : (
                laplacian_eigen.LaplacianEigen(n_clusters, n_nbrs),
                (gamma, )),
            "pvm" : (
                prototype_vector_machines.PVM(n_clusters, sigma2),
                (C1, C2)),
            "apg" : (
                anchor_points.AnchorPointsGMM(n_clusters, n_nbrs),
                (gamma, ))
            }
    
    X, Y, y = data[dataset]

    results = dict()
    ls, us = tools.random_data_split(X.shape[0], n_labeled, n_trials)
    logger = tools.get_logger(dataset + ".log")

    for alg in algs:

        logger.info("constructing %s model", alg)
        m = model[alg][0]
        m.fit(X)
        logger.info("%s model constructed", alg)
            
        results[alg] = []
        for trial in xrange(n_trials):

            l, u = ls[trial], us[trial]

            scores = m.predict(l, Y[l,:], *model[alg][1])
            y_hat = np.argmax(normalize(scores, axis=0, norm='l1'), axis=1)

            accuracy = 100.*np.sum(y_hat[u] == y[u])/len(u)

            logger.info("%s: sample %02d accuracy %.3f", alg, trial, accuracy)

            results[alg].append(accuracy)

    tools.print_formated_results(results)
