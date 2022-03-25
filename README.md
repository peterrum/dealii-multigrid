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
