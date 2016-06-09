#!/bin/env python3
from lennard_jones_potential import *
from multiprocessing import Pool
import numpy as np
import time

def main_3():
    "Same as before but calculate distances only once (using single CPU)."

    n_points = 2000

    print("Generating coordinates...", end='', flush=True)
    start = time.time()
    coords  = generate_coodinates(n_points, 3, upper=100, seed=5)
    stop = time.time()
    print("done!\t\tRuntime: {:6.3f} seconds".format(stop-start))

    print("Calculating distances...", end='', flush=True)
    start = time.time()
    distances   = calc_distances(coords)
    stop = time.time()
    print("done!\t\tRuntime: {:6.3f} seconds".format(stop-start))
    print(20*"-"+"\n")

    for n_CPUs in [1, 2, 4]:
        print("Start LJ-Potential on {} points using {} processes:".format(n_points, n_CPUs))

        print("Calculating LJ on {} CPUs ...".format(n_CPUs), end='', flush=True)
        start = time.time()
        pool = Pool(n_CPUs)                 # create multiprocessing pool
        results = pool.map(v_LJ, distances) # calculate results IN PARALLEL
        stop = time.time()
        print("done!\tRuntime: {:6.3f} seconds".format(stop-start))

        v_total = np.sum(results)           # sum up total

        print("v_Total: {:.2f}".format(v_total))# print result
        print(20*"-"+"\n")

if __name__ == '__main__':
    main_3()
