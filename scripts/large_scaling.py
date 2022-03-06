import json
import os
from argparse import ArgumentParser

def run_instance(counter, simulation, n_refinements, k, type):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["GeometryType"]        = simulation
    datastore["NRefGlobal"]          = n_refinements
    datastore["Degree"]              = k
    datastore["Type"]                = type

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def parseArguments():
    parser = ArgumentParser(description="Submit a simulation as a batch job")

    parser.add_argument('k', type=int)
    
    parser.add_argument("--hmg",  help="Run h-multigrid.",  action="store_true")
    parser.add_argument("--hpmg", help="Run h-pmultigrid.", action="store_true")

    parser.add_argument("--quadrant", help="Run quadrant test case.", action="store_true")
    parser.add_argument("--annulus", help="Run annulus test case.", action="store_true")
    
    arguments = parser.parse_args()
    return arguments

def main():
    options = parseArguments()

    k = options.k

    if options.quadrant:
        simulation = "quadrant"
    elif options.annulus:
        simulation = "annulus"
    else:
        raise ValueError("No simulation selected!")

    if options.hmg == options.hpmg:
        raise ValueError("You have to select either hmg or hpmg.")
        
    if options.hpmg:
        if options.k == 1:
            raise Exception("p-multigrid only working for k>1!")
    
    counter = 0

    for n_refinements in range(3,20):
        # local smoothing
        if options.hmg:
            run_instance(counter, simulation, n_refinements, k, "HMG-local")
            counter = counter + 1;
        else:
            run_instance(counter, simulation, n_refinements, k, "HPMG-local")
            counter = counter + 1;

        # global coarsening
        run_instance(counter, simulation, n_refinements, k, "HMG-global" if options.hmg else "HPMG")

        counter = counter + 1;


if __name__== "__main__":
  main()
