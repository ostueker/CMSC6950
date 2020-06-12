#!/bin/env python3

import numpy as np
import time
import sys

def v_LJ(r, A=5174175., B=27075.):
    "calculates Lennard-Jones potential for given distance r and A & B parameters"
    return ((A/(r**12))-(B/(r**6)))

def generate_coodinates(n=1000, dim=2, upper=100, seed=1):
    "generates n random coordinates in dim dimensional space"
    np.random.seed(seed)
    coords = np.random.rand(n, dim) * upper
    return coords

def calc_distances(points):
    "calculate distances for each pair of points"
    
    distances = []
    
    for i in range(len(points)):
        for j in range(len(points)):
            if j>i:
                d = np.sqrt(np.sum(np.square(np.subtract(points[i],points[j]))))
                distances.append(d)

    return np.array(distances)

def run_serial(n_points):
    ''' Run the calculation of the potential energy in serial.
    
    - generate coordinates for *n_points* of particles.
    - calculate the distances for particle pairs
    - calculate the LJ potential and sum it up.
    - print the total energy    
    '''
    print("Serial execution on {} points".format(n_points))
    start = time.time()

    coords  = generate_coodinates(n_points, 3, upper=100, seed=5)
    dists   = calc_distances(coords)

    v_total = 0
    for d in dists:
        v_total += v_LJ(d)
    stop = time.time()
    print("Runtime: {:.3f} seconds\n".format(stop-start))
    print("V_total = {:.1f}".format(v_total) )

def run_serial_timing(n_points):
    ''' Run the calculation of the potiential energy in serial.
    
    - generate coordinates for *n_points* of particles.
    - calculate the distances for particle pairs
    - calculate the LJ potential and sum it up.
    - print the total energy    
    '''
    print("Serial execution on {} points".format(n_points))
    start_total = time.time()

    print("Generating coordinates...", end='', flush=True)
    start = time.time()
    coords  = generate_coodinates(n_points, 3, upper=100, seed=5)
    stop = time.time()
    print("done!\tRuntime: {:.3f} seconds\n".format(stop-start))

    print("Calculating distances...", end='', flush=True)
    start = time.time()
    dists   = calc_distances(coords)
    stop = time.time()
    print("done!\tRuntime: {:.3f} seconds\n".format(stop-start))

    print("Calculating Potential...", end='', flush=True)
    start = time.time()
    v_total = 0
    for d in dists:
        v_total += v_LJ(d)
    stop = time.time()
    print("done!\tRuntime: {:.3f} seconds\n".format(stop-start))
    
    print("V_total = {:.1f}".format(v_total) )
    stop_total = time.time()
    print("Total Runtime: {:.3f} seconds\n".format(stop_total-start_total))

if __name__ == "__main__":
    n_points = 2000
    if not "timing" in sys.argv:
        run_serial(n_points)
    else:
        run_serial_timing(n_points)
