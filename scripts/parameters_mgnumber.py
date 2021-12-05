import json
import os

def run_instance(counter, n_refinements, k, solver, mg_number):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/parameters_mgnumber.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["NRefGlobal"]     = n_refinements
    datastore["Degree"]         = k
    datastore["Type"]           = solver
    datastore["MGNumberType"] = mg_number

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():
    
    counter = 0;

    for n_refinements in range(3,20):
        for tolerance in ["double", "float"]:
            for k in [1, 4]:
                for solver in ["HMG-local", "HMG-global"]:
                    run_instance(counter, n_refinements, k, solver, tolerance)
                    counter = counter + 1;


if __name__== "__main__":
  main()
