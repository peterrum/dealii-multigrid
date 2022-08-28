# Efficient distributed matrix-free multigrid methods on locally refined meshes for FEM computations

This project contains benchmarks for different multigrid varients:
- geometric local smoothing
- geometric global coarsening
- polynomial global coarsening
- AMG (ML, BoomerAMG)

It leverages on the infrastructure of deal.II and is the basis of the publication:

```
@article{munch2022gc,
  title         = {Efficient distributed matrix-free multigrid methods on locally refined 
                   meshes for FEM computations},
  author        = {Munch, Peter and Heister, Timo and Prieto Saavedra, Laura and 
                   Kronbichler, Martin},
  year          = {2022},
  archivePrefix = {arXiv:2203.12292},
}
```

### Getting started

Build `deal.II` with `p4est` (required), `Trilinos` (optional) and `PETSc` (optional) enabled (please specify the paths to those libraries):
```bash
git clone https://github.com/dealii/dealii.git
mkdir dealii-build
cd dealii-build/
echo "cmake \
    -D DEAL_II_WITH_64BIT_INDICES=\"OFF\" \
    -D CMAKE_BUILD_TYPE=\"DebugRelease\" \
    -D CMAKE_CXX_COMPILER=\"mpic++\" \
    -D CMAKE_CXX_FLAGS=\"-march=native -Wno-array-bounds  -std=c++17\" \
    -D DEAL_II_CXX_FLAGS_RELEASE=\"-O3\" \
    -D CMAKE_C_COMPILER=\"mpicc\" \
    -D DEAL_II_WITH_MPI=\"ON\" \
    -D DEAL_II_WITH_P4EST=\"ON\" \
    -D MPIEXEC_PREFLAGS=\"-bind-to none\" \
    -D DEAL_II_WITH_LAPACK=\"ON\" \
    -D DEAL_II_WITH_HDF5=\"OFF\" \
    -D DEAL_II_FORCE_BUNDLED_BOOST=\"OFF\" \
    -D DEAL_II_COMPONENT_DOCUMENTATION=\"OFF\" \
    -D P4EST_DIR=PATH_TO_P4EST \
    -D DEAL_II_WITH_TRILINOS=\"ON\" \
    -D TRILINOS_DIR=PATH_TO_TRILINOS \
    -D DEAL_II_WITH_PETSC:BOOL=\"ON\" \
    -D PETSC_DIR=PATH_TO_PETSC \
    -D PETSC_ARCH=\"arch-linux2-c-opt\" \
    ../dealii" > config.sh
. config.sh
make -j30
cd ..
```

Build the benchmarks:
```
git clone https://github.com/peterrum/dealii-multigrid.git
mkdir dealii-multigrid-build
cd dealii-multigrid-build
cmake ../dealii-multigrid -DDEAL_II_DIR=../dealii-build
make release
make -j10
cd ..
```

Run an experiment:

```
cd dealii-multigrid-build
mkdir -p small-scaling-quadrant
cd small-scaling-quadrant
python ../../dealii-multigrid/scripts/small-scaling.py quadrant
array=($(ls input_*.json));
mpirun -np 40 ../multigrid_throughput "${array[@]}"
cd ../..
```

This experiment runs, as an example, simulations on the octant mesh for different refinement levels. For each refinement level, the simulations are run for `LS, p=1`, `GC, p=1`, `LS, p=4`, and GC, p=4`. 

All experiments we have run are documented in the folder [experiments-skx](https://github.com/peterrum/dealii-multigrid/tree/master/experiments-skx).
