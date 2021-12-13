#!/bin/bash -l

#SBATCH -p regular
#SBATCH -N 15
#SBATCH -t 48:00:00
#SBATCH --account=desi
#SBATCH -J obiwan964
#SBATCH -o ./slurm_output/obiwan_%j.out
#SBATCH -L SCRATCH,project
#SBATCH -C haswell
#SBATCH --mail-user=kong.291@osu.edu  
#SBATCH --mail-type=ALL

# NERSC / Cray / Cori / Cori KNL things
export KMP_AFFINITY=disabled
export MPICH_GNI_FORK_MODE=FULLCOPY
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# Protect against astropy configs
export XDG_CONFIG_HOME=/dev/shm
srun -n $SLURM_JOB_NUM_NODES mkdir -p $XDG_CONFIG_HOME/astropy

export LEGACYVER=DR9.6.4
export rowstart=0
export name_for_run=decals_ngc
export dataset=normal
export nobj=200
export threads=16
#only need to be set while running cosmos 
#export cosmos_section=$1  
#export rsdir=rs${rowstart}_cosmos${cosmos_section}
export rsdir=more_rs${rowstart}
export TOT_SLICE=5
export SLICEI=0

export PYTHONPATH=./mpi:$PYTHONPATH

export topdir=/global/cfs/cdirs/desi/users/$USER/
export outdir=/global/cfs/cdirs/desi/users/$USER/$name_for_run/production_run/${rsdir}
mkdir -p $outdir


srun -N 15 -n 60 -c 16 shifter --module=mpich-cle6 --image=legacysurvey/legacypipe:DR9.6.4 python mpi_dr_slice.py
