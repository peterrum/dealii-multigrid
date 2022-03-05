# Test h-multigrid (quadrant or annulus, degree=1/4) for different cell weights.

import json
import os
import sys

def run_instance(counter, geometry_type, n_refinements, k, solver, partitioner):
    # read default settings
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["Type"]           = solver
    datastore["GeometryType"]   = geometry_type
    datastore["NRefGlobal"]     = n_refinements
    datastore["Degree"]         = k

    if partitioner != "":
      datastore["PartitionerName"] = partitioner

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():

    geometry_type = sys.argv[1]
    degree        = sys.argv[2]
    
    if geometry_type == "quadrant":
        min_ref = 3;
    elif geometry_type == "annulus":
        min_ref = 5;
    else:
        print('Geometry type not known!')

    solvers = [["HMG-local",""], ["HMG-global", ""], ["HMG-global", "FirstChildPolicy-2.0"]]

    counter = 0

    for n_refinements in range(min_ref,20):            # number of refinements
        for k in [degree]:                             # degree
            for [solver, partitioner_type] in solvers: # h-multigrid types
                run_instance(counter, geometry_type, n_refinements, k, solver, partitioner_type)
                counter = counter + 1;


if __name__== "__main__":
  main()
