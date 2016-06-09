#!/bin/env python3
from lennard_jones_potential import *
from multiprocessing import Pool
import numpy as np
import time

def main_1():
    "Calculate LJ energy of 2000 points using 1, 2, 4 CPUs."

    n_points = 2000

    for n_CPUs in [1, 2, 4]:

        print("Parallel execution of {} points using {} processes".format(n_points, n_CPUs))
        start = time.time()                 # start timer

        coords = generate_coodinates(n_points, 3, upper=100, seed=5)

        distances = calc_distances(coords)

        pool = Pool(n_CPUs)                 # create multiprocessing pool
        results = pool.map(v_LJ, distances) # calculate results IN PARALLEL

        v_total = np.sum(results)           # sum up total
        print("v_Total: {}".format(v_total))# print result

        stop = time.time()                  # stop timer
        print("Total Runtime: {:.3f} seconds".format(stop-start))
        print(20*"-"+"\n")

if __name__ == '__main__':
    main_1()