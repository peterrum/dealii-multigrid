import json
import argparse, os
import math
import subprocess
import shutil
import sys 

cmd = """#!/bin/bash
# Job Name and Files (also --job-name)
#SBATCH -J LIKWID
#Output and error (also --output, --error):
#SBATCH -o node-{1}.out
#SBATCH -e node-{1}.e
#Initial working directory (also --chdir):
#SBATCH -D ./
#Notification and type
#SBATCH --mail-type=END
#SBATCH --mail-user=peter.muench@tum.de
# Wall clock limit:
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#Setup of execution environment
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --account=pr83te
#
## #SBATCH --switches=4@24:00:00
#SBATCH --partition={2}
#Number of nodes and MPI tasks per node:
#SBATCH --nodes={0}
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#module list
#source ~/.bashrc
# lscpu

#module unload mkl mpi.intel intel
#module load intel/19.0 mkl/2019
#module load gcc/9
#module unload mpi.intel
#module load mpi.intel/2019_gcc
#module load cmake
#module load slurm_setup

module unload intel-mpi/2019-intel
module unload intel/19.0.5
module load gcc/9
module load intel-mpi/2019-gcc

pwd

array=($(ls *.json))
mpirun -np {3} ../multigrid_throughput_params \"${{array[@]}}\"
"""

def main():

    max_nodes = 1000000

    if len(sys.argv[1]) > 1:
      max_nodes = int(sys.argv[1])

    for n in [ a for a in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072] if a <= max_nodes]:

        label = ""
        if n <= 16:
            label = "micro"
        elif n <= 768:
            label = "general"
        elif n <= 3072:
            label = "large"

        with open("node-%s.cmd" % (str(n).zfill(4)), 'w') as f:
            f.write(cmd.format(str(n), str(n).zfill(4), label, 48*n))

if __name__== "__main__":
  main()
