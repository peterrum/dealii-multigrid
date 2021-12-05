import json
import os

def run_instance(counter, n_refinements, k, solver, tolrance):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/parameters_tolerance.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["NRefGlobal"]     = n_refinements
    datastore["Degree"]         = k
    datastore["Type"]           = solver
    datastore["RelativeTolerance"] = tolrance

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():
    
    counter = 0;

    for n_refinements in range(3,20):
        for tolerance in [1e-4, 1e-6, 1e-8, 1e-10]:
            for k in [1, 4]:
                for solver in ["HMG-local", "HMG-global"]:
                    run_instance(counter, n_refinements, k, solver, tolerance)
                    counter = counter + 1;


if __name__== "__main__":
  main()
