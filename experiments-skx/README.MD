# General commands
```bash
for cmd in *.cmd; do sbatch $cmd; done
```

# Section 4 and Section 5

## Local smoothing vs. global coarsening

### Small scale

```bash
mkdir -p small-scaling-annulus
cd small-scaling-annulus
python ../../dealii-multigrid/scripts/small-scaling.py annulus
sbatch ../../dealii-multigrid/experiments-skx/small-scaling-annulus-0001.cmd
sbatch ../../dealii-multigrid/experiments-skx/small-scaling-annulus-0192.cmd
cd ..

mkdir -p small-scaling-quadrant
cd small-scaling-quadrant
python ../../dealii-multigrid/scripts/small-scaling.py quadrant
sbatch --time=2:00:00 ../../dealii-multigrid/experiments-skx/small-scaling-quadrant-0001.cmd
sbatch ../../dealii-multigrid/experiments-skx/small-scaling-quadrant-0192.cmd
cd ..

mkdir -p small-scaling-hypercube
cd small-scaling-hypercube
python ../../dealii-multigrid/scripts/small-scaling-hypercube.py
sbatch ../../dealii-multigrid/experiments-skx/parameters-0768.cmd
cd ..

mkdir -p small-scaling-gaussian
cd small-scaling-gaussian
python ../../dealii-multigrid/scripts/small-scaling-gaussian.py
sbatch ../../dealii-multigrid/experiments-skx/parameters-0768.cmd
cd ..
```

### Large scale

```
mkdir -p large-scaling-annulus-1
cd large-scaling-annulus-1
python ../../dealii-multigrid/scripts/large_scaling.py 1 --hmg --annulus
python ../../dealii-multigrid/experiments-skx/large-scaling-create-job-files.py
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir -p large-scaling-annulus-4
cd large-scaling-annulus-4
python ../../dealii-multigrid/scripts/large_scaling.py 4 --hmg --annulus
python ../../dealii-multigrid/experiments-skx/large-scaling-create-job-files.py
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir -p large-scaling-quadrant-1
cd large-scaling-quadrant-1
python ../../dealii-multigrid/scripts/large_scaling.py 1 --hmg --quadrant
python ../../dealii-multigrid/experiments-skx/large-scaling-create-job-files.py
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir -p large-scaling-quadrant-4
cd large-scaling-quadrant-4
python ../../dealii-multigrid/scripts/large_scaling.py 4 --hmg --quadrant
python ../../dealii-multigrid/experiments-skx/large-scaling-create-job-files.py
for cmd in *.cmd; do sbatch $cmd; done
cd ..
```

## Global coarsening with first child policy

### Small scale

```bash
mkdir -p small-scaling-gc-fcp-annulus
cd small-scaling-gc-fcp-annulus
python ../../dealii-multigrid/scripts/small-scaling.py annulus FirstChildPolicy-2.0
sbatch ../../dealii-multigrid/experiments-skx/small-scaling-annulus-0192.cmd
cd ..

mkdir -p small-scaling-gc-fcp-quadrant
cd small-scaling-gc-fcp-quadrant
python ../../dealii-multigrid/scripts/small-scaling.py quadrant FirstChildPolicy-2.0
sbatch ../../dealii-multigrid/experiments-skx/small-scaling-quadrant-0192.cmd
cd ..
```

### Large scale: quadrant

```bash
mkdir -p large-scaling-quadrant-1-gc-fcp
cd large-scaling-quadrant-1-gc-fcp
python ../../dealii-multigrid/scripts/large_scaling_partitioning.py quadrant 1
python ../../dealii-multigrid/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir -p large-scaling-quadrant-4-gc-fcp
cd large-scaling-quadrant-4-gc-fcp
python ../../dealii-multigrid/scripts/large_scaling_partitioning.py quadrant 4
python ../../dealii-multigrid/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..
```

### Large scale: annulus (optional)

```bash
mkdir -p large-scaling-annulus-1-gc-fcp
cd large-scaling-annulus-1-gc-fcp
python ../../dealii-multigrid/scripts/large_scaling_partitioning.py annulus 1
python ../../dealii-multigrid/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..

mkdir -p large-scaling-annulus-4-gc-fcp
cd large-scaling-annulus-4-gc-fcp
python ../../dealii-multigrid/scripts/large_scaling_partitioning.py annulus 4
python ../../dealii-multigrid/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..
```

# Section 6

```
mkdir large-scaling-quadrant-hp
cd large-scaling-quadrant-hp
python ../../dealii-multigrid/scripts/large_scaling_hp.py 4 --hpmg --quadrant
python ../../dealii-multigrid/experiments-skx/large-scaling-create-job-files.py 512
for cmd in *.cmd; do sbatch $cmd; done
cd ..
```

# Appendix to Section 5

```bash
mkdir -p parameters-smoother-degree
cd parameters-smoother-degree
python ../../dealii-multigrid/scripts/parameters_smoother_degree.py
sbatch ../../dealii-multigrid/experiments-skx/parameters-0768.cmd
cd ..

mkdir -p parameters-mgnumber
cd parameters-mgnumber
python ../../dealii-multigrid/scripts/parameters_mgnumber.py
sbatch ../../dealii-multigrid/experiments-skx/parameters-0768.cmd
cd ..

mkdir -p parameters-tolerance
cd parameters-tolerance
python ../../dealii-multigrid/scripts/parameters_tolerance.py
sbatch ../../dealii-multigrid/experiments-skx/parameters-0768.cmd
cd ..

mkdir -p parameters-weights
cd parameters-weights
python ../../dealii-multigrid/scripts/parameters_weights.py
sbatch ../../dealii-multigrid/experiments-skx/parameters-0768.cmd
sbatch ../../dealii-multigrid/experiments-skx/parameters-24576.cmd
cd ..
```

# Appendix to Section 6

```bash
mkdir -p parameters-hp-amg
cd parameters-hp-amg
python ../../dealii-multigrid/scripts/parameters_hp_amg.py
sbatch ../../dealii-multigrid/experiments-skx/parameters-0768.cmd
cd ..
```