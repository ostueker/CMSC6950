#!/bin/env python3
from lennard_jones_potential import *
from multiprocessing import Pool
import numpy as np
import time

def generate_pairs(n):
    "generate unique point pairs (i<j)"
    pairs = []
    for i in range(n):
        for j in range(n):
            if j>i:
                pairs.append([i,j])
    return pairs

def calc_pot(args):
    "calculate the distance for a pair i j in coords and return LJ potential"

    i, j, coords = args                 # unpack arguments tupel
    r = np.sqrt(np.sum(np.square(np.subtract(coords[i],coords[j]))))
    v = v_LJ(r)

    return v

def main_4():
    "Calculate distances in parallel as well."
    n_points = 2000

    print("Generating coordinates...", end='', flush=True)
    start = time.time()
    coords  = generate_coodinates(n_points, 3, upper=100, seed=5)
    stop = time.time()
    print("done!\t\tRuntime: {:6.3f} seconds".format(stop-start))

    print("Generating pairs...", end='', flush=True)
    start = time.time()
    pairs   = generate_pairs(len(coords))
    stop = time.time()
    print("done!\t\tRuntime: {:6.3f} seconds".format(stop-start))

    for n_CPUs in [1, 2, 4]:
        print("Parallel LJ-Potential on {} points using {} processes:".format(n_points, n_CPUs))
        pool = Pool(n_CPUs)

        print("Calculating LJ on {} CPUs ...".format(n_CPUs), end='', flush=True)
        start = time.time()

        tasks = []                              # prepare tasks
        for p in pairs:
            args = (p[0], p[1], coords)         # pack arguments tupel
            tasks.append(args)

        results = pool.map(calc_pot, tasks)     # <-- in parallel

        stop = time.time()
        print("done!\tRuntime: {:6.3f} seconds".format(stop-start))

        v_total = np.sum(results)               # sum up total
        print("v_Total: {}".format(v_total))    # print result
        print(20*"-"+"\n")

if __name__ == '__main__':
    main_4()
