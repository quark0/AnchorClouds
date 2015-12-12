import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import time
import manifold_generator
import tools
import prototype_vector_machines
import laplacian_eigen
import anchor_points

if __name__ == '__main__':
    from anchor_clouds import anchor_clouds

    np.random.seed(1267)
    dataset     = 'swiss'
    n_trials    = 20
    visualize   = False

    #MNIST
    if dataset == 'mnist':
        X, Y, y = manifold_generator.mnist()
        n_nbrs      = 3
        n_clusters  = 1000
        n_labeled   = 100
        inner_dim   = 6
        gamma       = 1e-1
        n_data_per_anchor = 50
        algs = ["ap", "ac"]

    #Letter.scale
    elif dataset == 'letter':
        X, Y, y = manifold_generator.letter() 
        n_nbrs      = 3
        n_clusters  = 512
        n_labeled   = 256
        inner_dim   = 4
        gamma       = 1e-2
        n_data_per_anchor = 50
        algs = ["ap", "ac", "nn", "pvm"]

    #USPS
    elif dataset == 'usps':
        X, Y, y = manifold_generator.usps() 
        n_nbrs      = 2
        n_clusters  = 200
        n_labeled   = 100
        inner_dim   = 5
        gamma       = 1e-1
        n_data_per_anchor = 40
        algs = ["ap", "ac", "nn"]

    #Double Swiss Roll
    elif dataset == 'swiss':
        X, Y, y = manifold_generator.double_swiss_roll(n_samples=10000, var=.8)
        n_nbrs      = 3
        n_clusters  = 48
        n_labeled   = 24
        inner_dim   = 1
        gamma       = 1e-2
        n_data_per_anchor = 200
        # heuristics: keep m = O(n/d)
        algs = ["ap", "ac", "nn", "pvm", "apg"]

    n = X.shape[0]
    #tools.visualize_datapoints(X, y, "Ground Truth")

    ls, us = tools.random_data_split(n, n_labeled, n_trials)

    results = {}
    for alg in algs:

        print alg

        t_start = time.time()
        if alg == "ap":
            ap = anchor_points.AnchorPoints(n_clusters, n_nbrs)
            ap.fit(X)
        if alg == "apg":
            ap_gmm = anchor_points.AnchorPointsGMM(n_clusters, n_nbrs)
            ap_gmm.fit(X)
        if alg == "ac":
            A, Z = anchor_clouds(X, inner_dim, n_clusters, n_data_per_anchor, n_nbrs)
        if alg == "le":
            le = laplacian_eigen.LaplacianEigen(n_clusters, n_nbrs)
            le.fit(X)
        if alg == "nn":
            nn = KNeighborsClassifier(n_neighbors=1)
        if alg == "pvm":
            sigma2, C1, C2 = 1, 1, 1
            pvm = prototype_vector_machines.PVM(n_clusters, sigma2)
            pvm.fit(X)

        if visualize == True and alg.startswith("anchor"):
            tools.visualize_edges(X, A, Z, 1e-6, alg)
            
        t_elapsed = time.time() - t_start
        print '%.3f secs' % t_elapsed

        results[alg] = []
        for trial in xrange(n_trials):

            print "%d" % trial
            l, u = ls[trial], us[trial]

            if alg == "ap":
                scores = ap.predict(l, Y[l,:], gamma)
            elif alg == "apg":
                scores = ap_gmm.predict(l, Y[l,:], gamma)
            elif alg == "ac":
                scores = anchor_points.reduced_sml(Z, l, Y[l,:], gamma)
            elif alg == "le":
                scores = le.predict(l, Y[l,:], gamma)
            elif alg == "nn":
                scores = nn.fit(X[l,:], Y[l,:]).predict(X)
            elif alg == "pvm":
                scores = pvm.predict(l, Y[l,:], C1, C2)

            y_hat = np.argmax(normalize(scores, axis=0, norm='l1'), axis=1)

            acc = 100.*np.sum(y_hat[u] == y[u])/len(u)
            results[alg].append(acc)

    tools.print_formated_results(results)
