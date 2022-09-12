# Stokes Flow Mantle Convection example

## Setup

Please install deal.II with MPI, p4est, and Trilinos. We used v9.4.0 but a recent master should also work.

Then configure and compile ASPECT from the branch chosen in the git submodule in this directory.

## The testcase

Go to ./aspect/benchmarks/nsinker_spherical_shell/ and configure and compile:
```
cd aspect/benchmarks/nsinker_spherical_shell/
cmake -D ASPECT_DIR= .
make
```

You can then run from the same directory using
```
mpirun -n 24 ./aspect test.prm
```

## More information

The plugin that defines the viscosity and density including the sinker locations is at ./aspect/benchmarks/nsinker_spherical_shell/nsinker.cc but the benchmark is also available in the main version of ASPECT [here](https://github.com/geodynamics/aspect/tree/main/benchmarks/nsinker_spherical_shell).
