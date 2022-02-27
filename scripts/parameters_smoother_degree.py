# Test h-multigrid and AMG (quadrant, degree=1/4) for different Cheyshev smoother degrees.

import json
import os

def run_instance(counter, n_refinements, k, solver, degree):
    # read default settings
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["Type"]           = solver
    datastore["NRefGlobal"]     = n_refinements
    datastore["Degree"]         = k
    datastore["SmootherDegree"] = degree

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():
    
    counter = 0;

    for n_refinements in range(3,20):                         # number of refinements
        for k in [1, 4]:                                      # degree
            for solver in ["HMG-local", "HMG-global", "AMG"]: # multigrid type
                for degree in [3, 6]:                         # smoothing degree
                    if solver != "AMG" or (k==1 and degree == 3):
                        # note: run AMG only once with linear elements
                        run_instance(counter, n_refinements, k, solver, degree)
                        counter = counter + 1;

if __name__== "__main__":
  main()
