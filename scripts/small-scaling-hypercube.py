# Test h-multigrid (hypercube, degree=4).

import json
import os
import sys

def run_instance(counter, n_refinements, k, solver):
    # read default settings
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["Type"]            = solver
    datastore["GeometryType"]    = "hypercube"
    datastore["NRefGlobal"]      = n_refinements
    datastore["Degree"]          = k
    datastore["PartitionerName"] = "FirstChildPolicy-2.0"

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():

    counter = 0

    for n_refinements in range(3,20):                  # number of refinements
        for k in [1, 4]:                               # degree
            for solver in ["HMG-local", "HMG-global"]: # h-multigrid types
                run_instance(counter, n_refinements, k, solver)
                counter = counter + 1;

if __name__== "__main__":
  main()
