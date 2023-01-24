#!/bin/bash

# JobName #
#PBS -N JmpiHJacobi

# Which Queue to use #
#PBS -q N10C80

# Max Wall time #
#PBS -l walltime=00:20:00

# How many nodes and tasks per node
#PBS -l select=8:ncpus=8:mpiprocs=2:ompthreads=4:mem=16400000kb

#Change Working directory to SUBMIT directory
cd $PBS_O_WORKDIR

# Run executable #
mpirun jacobi_mpiH.x < input
