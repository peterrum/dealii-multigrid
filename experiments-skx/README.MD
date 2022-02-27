# Section 4 and Section 5

## Local smoothing vs. global coarsening

### Small scale

```bash
mkdir small-scaling-annulus
cd small-scaling-annulus
python ../../dealii-multigrid/scripts/small-scaling.py annulus
sbatch ../../dealii-multigrid/experiments-skx/small-scaling-annulus-0001.cmd
sbatch ../../dealii-multigrid/experiments-skx/small-scaling-annulus-0192.cmd
cd ..

mkdir small-scaling-quadrant
cd small-scaling-quadrant
python ../../dealii-multigrid/scripts/small-scaling.py quadrant
sbatch ../../dealii-multigrid/experiments-skx/small-scaling-quadrant-0001.cmd
sbatch ../../dealii-multigrid/experiments-skx/small-scaling-quadrant-0192.cmd
cd ..
```

```bash
mkdir small-scaling-hypercube
cd small-scaling-hypercube
python ../../dealii-multigrid/scripts/small-scaling-hypercube.py
sbatch ../../dealii-multigrid/experiments-skx/parameters-0768.cmd
cd ..
```

### Large scale

```
mkdir large-scaling-annulus
cd large-scaling-annulus
python ../../dealii-multigrid/scripts/large_scaling.py --hmg --annulus
python ../../dealii-multigrid/experiments-skx/large-scaling-create-job-files.py
sbatch node-0001.cmd
sbatch node-0002.cmd
sbatch node-0004.cmd
sbatch node-0008.cmd
sbatch node-0016.cmd
sbatch node-0032.cmd
sbatch node-0064.cmd
sbatch node-0128.cmd
sbatch node-0256.cmd
sbatch node-0512.cmd
sbatch node-1024.cmd
sbatch node-2048.cmd
sbatch node-3096.cmd
cd ...
```

```
mkdir large-scaling-quadrant
cd large-scaling-quadrant
python ../../dealii-multigrid/scripts/large_scaling.py --hmg --quadrant
python ../../dealii-multigrid/experiments-skx/large-scaling-create-job-files.py
sbatch node-0001.cmd
sbatch node-0002.cmd
sbatch node-0004.cmd
sbatch node-0008.cmd
sbatch node-0016.cmd
sbatch node-0032.cmd
sbatch node-0064.cmd
sbatch node-0128.cmd
sbatch node-0256.cmd
sbatch node-0512.cmd
sbatch node-1024.cmd
sbatch node-2048.cmd
sbatch node-3096.cmd
cd ...
```

## Global coarsening with first child policy

```bash
mkdir small-scaling-gc-fcp-annulus
cd small-scaling-gc-fcp-annulus
python ../../dealii-multigrid/scripts/small-scaling.py annulus FirstChildPolicy-2.0
sbatch ../../dealii-multigrid/experiments-skx/small-scaling-annulus-0192.cmd
cd ..

mkdir small-scaling-gc-fcp-quadrant
cd small-scaling-gc-fcp-quadrant
python ../../dealii-multigrid/scripts/small-scaling.py quadrant FirstChildPolicy-2.0
sbatch ../../dealii-multigrid/experiments-skx/small-scaling-quadrant-0192.cmd
cd ..

mkdir small-scaling-gc-default-annulus
cd small-scaling-gc-default-annulus
python ../../dealii-multigrid/scripts/small-scaling.py annulus DefaultPolicy
sbatch ../../dealii-multigrid/experiments-skx/small-scaling-annulus-0192.cmd
cd ..

mkdir small-scaling-gc-default-quadrant
cd small-scaling-gc-default-quadrant
python ../../dealii-multigrid/scripts/small-scaling.py quadrant DefaultPolicy
sbatch ../../dealii-multigrid/experiments-skx/small-scaling-quadrant-0192.cmd
cd ..
```

# Section 6

```
mkdir large-scaling-quadrant-hp
cd large-scaling-quadrant-hp
python ../../dealii-multigrid/scripts/large_scaling.py --hpmg --quadrant
python ../../dealii-multigrid/experiments-skx/large-scaling-create-job-files.py
sbatch node-0001.cmd
sbatch node-0002.cmd
sbatch node-0004.cmd
sbatch node-0008.cmd
sbatch node-0016.cmd
sbatch node-0032.cmd
sbatch node-0064.cmd
sbatch node-0128.cmd
sbatch node-0256.cmd
sbatch node-0512.cmd
sbatch node-1024.cmd
sbatch node-2048.cmd
sbatch node-3096.cmd
cd ...
```

# Appendix to Section 5

```bash
mkdir parameters-smoother-degree
cd parameters-smoother-degree
python ../../dealii-multigrid/scripts/parameters_smoother_degree.py
sbatch ../../dealii-multigrid/experiments-skx/parameters-0768.cmd
cd ..

mkdir parameters-mgnumber
cd parameters-mgnumber
python ../../dealii-multigrid/scripts/parameters_mgnumber.py
sbatch ../../dealii-multigrid/experiments-skx/parameters-0768.cmd
cd ..

mkdir parameters-tolerance
cd parameters-tolerance
python ../../dealii-multigrid/scripts/parameters_tolerance.py
sbatch ../../dealii-multigrid/experiments-skx/parameters-0768.cmd
cd ..

mkdir parameters-weights
cd parameters-weights
python ../../dealii-multigrid/scripts/parameters_weights.py
sbatch ../../dealii-multigrid/experiments-skx/parameters-0768.cmd
sbatch ../../dealii-multigrid/experiments-skx/parameters-24576.cmd
cd ..
```

# Appendix to Section 6


```bash
mkdir parameters-hp-amg
cd parameters-hp-amg
python ../../dealii-multigrid/scripts/parameters_hp_amg.py
sbatch ../../dealii-multigrid/experiments-skx/parameters-0768.cmd
cd ..
```