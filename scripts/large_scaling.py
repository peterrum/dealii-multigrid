import json
import os
from argparse import ArgumentParser

def run_instance(counter, simulation, n_refinements, k, type, policy, max_level):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/large_scaling.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["GeometryType"]        = simulation
    datastore["NRefGlobal"]          = n_refinements
    datastore["Degree"]              = k
    datastore["Type"]                = type
    datastore["SmootherDegree"]      = 3
    datastore["PartitionerName"]     = policy
    datastore["MinLevel"]            = max_level
    datastore["CoarseSolverNCycles"] = 2

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def parseArguments():
    parser = ArgumentParser(description="Submit a simulation as a batch job")
    
    parser.add_argument("--hmg",  help="Run h-multigrid.",  action="store_true")
    parser.add_argument("--hpmg", help="Run h-pmultigrid.", action="store_true")

    parser.add_argument("--quadrant", help="Run quadrant test case.", action="store_true")
    parser.add_argument("--annulus", help="Run annulus test case.", action="store_true")
    
    arguments = parser.parse_args()
    return arguments

def main():
    options = parseArguments()

    if options.quadrant:
        simulation = "quadrant"
    elif options.annulus:
        simulation = "annulus"
    else:
        raise ValueError("No simulation selected!")

    if options.hmg == options.hpmg:
        raise ValueError("You have to select either hmg or hpmg.")
    
    counter = 0;

    for n_refinements in range(3,20):
        
        if options.hmg:
            ks = [1, 4]
        else:
            ks = [4]

        for k in ks:
            # local smoothing
            if options.hmg:
                run_instance(counter, simulation, n_refinements, k, "HMG-local",  "DefaultPolicy", 0)
                counter = counter + 1;
            else:
                for policy in ["DefaultPolicy", "CellWeightPolicy-1.0", "CellWeightPolicy-1.5", "CellWeightPolicy-2.0", "CellWeightPolicy-2.5", "CellWeightPolicy-3.0", "FirstChildPolicy"]:
                    run_instance(counter, simulation, n_refinements, k, "HPMG-local",  policy, 0)
                    counter = counter + 1;

            # global coarsening
            for policy in ["DefaultPolicy", "CellWeightPolicy-1.0", "CellWeightPolicy-1.5", "CellWeightPolicy-2.0", "CellWeightPolicy-2.5", "CellWeightPolicy-3.0", "FirstChildPolicy"]:
                run_instance(counter, simulation, n_refinements, k, "HMG-global" if options.hmg else "HPMG",  policy, 0)
                counter = counter + 1;

            ## HP-multigrid + AMG
            #if options.hpmg:
            #    run_instance(counter, simulation, n_refinements, k, "PMG",  "DefaultPolicy", n_refinements + 1)
            #    counter = counter + 1;


if __name__== "__main__":
  main()
