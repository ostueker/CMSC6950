#!/bin/env python3

from lennard_jones_potential import *
from multiprocessing import Pool
import numpy as np

def generate_pairs(n):
    pairs = []
    
    for i in range(n):
        for j in range(n):
            if j>i:
                pairs.append([i,j])
    return pairs



def calc_pot(args):
    "calculate the distance for a pair i j in coords and return LJ potential"

    i, j, points = args

    r = np.sqrt(np.sum(np.square(np.subtract(points[i],points[j]))))

    return v_LJ(r)


if __name__ == '__main__':
    n_points = 2000

    print("Generating coordinates...", end='', flush=True)
    start = time.time()
    coords  = generate_coodinates(n_points, 3, upper=100, seed=5)
    stop = time.time()
    print("done!\tRuntime: {:.3f} seconds\n".format(stop-start))


    print("Generating pairs...", end='', flush=True)
    start = time.time()
    pairs   = generate_pairs(len(coords))
    stop = time.time()
    print("done!\tRuntime: {:.6f} seconds\n".format(stop-start))


    for n_CPUs in [1, 2, 4, 8]:
        pool = Pool(n_CPUs)

        print("Parallel LJ-Potential on {} points using {} processes".format(n_points, n_CPUs))
        start = time.time()

        tasks = [ (p[0], p[1], coords) for p in pairs]
        results = pool.map(calc_pot, tasks)

        v_total = np.sum(results)
        print(v_total)

        stop = time.time()
        print("Runtime: {:.3f} seconds\n".format(stop-start))

