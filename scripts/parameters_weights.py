# Test h-multigrid (quadrant, degree=1/4) for different cell weights.

import json
import os

def run_instance(counter, n_refinements, k, weight):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["Type"]            = "HMG-global"
    datastore["NRefGlobal"]      = n_refinements
    datastore["Degree"]          = k
    datastore["PartitionerName"] = "CellWeightPolicy-%f" % weight

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():
    
    counter = 0;

    for n_refinements in range(3,20):                      # number of refinements
        for k in [1, 4]:                                   # degree
            for weight in [1.0, 1.5, 2.0, 2.5, 3.0]:       # weight
                run_instance(counter, n_refinements, k,weight)
                counter = counter + 1;


if __name__== "__main__":
  main()
