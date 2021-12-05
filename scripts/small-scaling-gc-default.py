import json
import os
import sys

def run_instance(counter, geometry_type, n_refinements, k, solver):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/small-scaling-gc-default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["GeometryType"]   = geometry_type
    datastore["NRefGlobal"]     = n_refinements
    datastore["Degree"]         = k
    datastore["Type"]           = solver
    datastore["SmootherDegree"] = 3

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():

    if len(sys.argv) != 2:
        print('You have specified too many arguments')
        sys.exit()

    geometry_type = sys.argv[1]
    
    if geometry_type == "quadrant":
        min_ref = 3;
    elif geometry_type == "annulus":
        min_ref = 5;
    else:
        print('Geometry type not known!')

    counter = 0

    for n_refinements in range(min_ref,20):
        for k in [1, 4]:
            for solver in ["HMG-global"]:
                run_instance(counter, geometry_type, n_refinements, k, solver)
                counter = counter + 1;


if __name__== "__main__":
  main()
