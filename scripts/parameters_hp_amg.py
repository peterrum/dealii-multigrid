# Test p-multigrid (quadrant, degree=4) with different coarse-grid solvers.

import json
import os

def run_instance(counter, t, n_refinements, min_level, coarse_grid_solver_type, n_cyles):
    with open(os.path.dirname(os.path.abspath(__file__)) + "/default.json", 'r') as f:
       datastore = json.load(f)

    # make modifications
    datastore["Type"]                 = t
    datastore["GeometryType"]         = "quadrant"
    datastore["NRefGlobal"]           = n_refinements
    datastore["Degree"]               = 4
    datastore["MinLevel"]             = min_level
    datastore["CoarseGridSolverType"] = coarse_grid_solver_type
    datastore["CoarseSolverNCycles"]  = n_cyles

    # write data to output file
    with open("./input_%s.json" % (str(counter).zfill(4)), 'w') as f:
        json.dump(datastore, f, indent=4, separators=(',', ': '))

def main():
    
    counter = 0;

    for n_refinements in range(3,20): # number of refinements

        # local smoothing
        run_instance(counter, "HPMG-local", n_refinements, 0, "amg", 1)
        counter = counter + 1;

        # global coarsening
        run_instance(counter, "HPMG", n_refinements, 0, "amg", 1)
        counter = counter + 1;

        # Trilinos ML with different number of repetitions
        for k in range(1, 5):
            run_instance(counter, "HPMG", n_refinements, n_refinements + 1, "amg", k)
            counter = counter + 1;

        # PETSc BoomerAMG
        run_instance(counter, "HPMG", n_refinements, n_refinements + 1, "amg_petsc", 2)
        counter = counter + 1;

if __name__== "__main__":
  main()