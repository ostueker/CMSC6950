#!/bin/env python3

from lennard_jones_potential import *
from multiprocessing import Pool
import numpy as np


if __name__ == '__main__':
    n_points = 2000

    print("Generating coordinates...", end='', flush=True)
    start = time.time()
    coords  = generate_coodinates(n_points, 3, upper=100, seed=5)
    stop = time.time()
    print("done!\tRuntime: {:.3f} seconds\n".format(stop-start))

    print("Calculating distances...", end='', flush=True)
    start = time.time()
    distances   = calc_distances(coords)
    stop = time.time()
    print("done!\tRuntime: {:.6f} seconds\n".format(stop-start))

    for n_CPUs in [1, 2, 4, 8]:
        pool = Pool(n_CPUs)

        print("Parallel LJ-Potential on {} points using {} processes".format(n_points, n_CPUs))
        start = time.time()

        results = pool.map(v_LJ, distances)
        v_total = np.sum(results)
        print(v_total)

        stop = time.time()
        print("Runtime: {:.3f} seconds\n".format(stop-start))

