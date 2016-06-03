#!/bin/env python3

from lennard_jones_potential import *
from multiprocessing import Pool
import numpy as np

def run_parallel(n_points, n_CPUs=1):
    ''' Run the calculation of the potiential energy in serial.
    
    - generate coordinates for *n_points* of particles.
    - calculate the distances for particle pairs
    - calculate the LJ potential and sum it up.
    - print the total energy    
    '''

    pool = Pool(n_CPUs)

    coords  = generate_coodinates(n_points, 3, upper=100, seed=5)
    dists   = calc_distances(coords)

    results = pool.map(v_LJ, dists)
    v_total = np.sum(results)
    
    print(v_total)


if __name__ == '__main__':
    for n_CPUs in [1, 2, 4, 8]:
        n_points = 2000
        print("Parallel execution of {} points using {} processes".format(n_points, n_CPUs))
        start = time.time()
        run_parallel(n_points, n_CPUs)
        stop = time.time()
        print("Runtime: {:.3f} seconds\n".format(stop-start))

