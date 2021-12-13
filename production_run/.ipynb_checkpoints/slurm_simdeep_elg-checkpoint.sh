#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 15
#SBATCH -t 00:30:00
#SBATCH --account=desi
#SBATCH -J obiwan
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
#srun -n $SLURM_JOB_NUM_NODES mkdir -p $XDG_CONFIG_HOME/astropy

export rowstart=0
export name_for_run=sim_deep  
export dataset=deep
export nobj=500
export threads=64
#only need to be set while running cosmos 
#export cosmos_section=$1  
#export rsdir=rs${rowstart}_cosmos${cosmos_section}
export rsdir=rs0

export PYTHONPATH=./mpi:$PYTHONPATH

export topdir=$CSCRATCH/Obiwan/dr9_LRG/obiwan_out/
export outdir=$CSCRATCH/Obiwan/dr9_LRG/obiwan_out/$name_for_run/output/${rsdir}
mkdir -p $outdir



#srun -N 1 -n 1 -c 64 shifter --module=mpich-cle6 --image=driftingpig/obiwan:dr9.3 ./run.sh
srun -N 22 -n 44 -c 32 shifter --module=mpich-cle6 --image=legacysurvey/legacypipe:DR9.7.1 python mpi_elg.py
