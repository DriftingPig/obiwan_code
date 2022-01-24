#!/bin/bash -l

#SBATCH -p regular
#SBATCH -N 15
#SBATCH -t 15:00:00
#SBATCH --account=desi
#SBATCH -J o965
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

export LEGACYVER=DR9.6.5
export rowstart=0
export name_for_run=decals_sgc
export dataset=normal
export nobj=200
export threads=16
#only need to be set while running cosmos 
#export cosmos_section=$1  
#export rsdir=rs${rowstart}_cosmos${cosmos_section}
export rsdir=more_rs${rowstart}

export PYTHONPATH=./mpi:$PYTHONPATH

export topdir=/global/cfs/cdirs/desi/users/huikong/
#export outdir=$CSCRATCH/Obiwan/dr9_LRG/obiwan_out/$name_for_run/output/${rsdir}
export outdir=/global/cfs/cdirs/desi/users/huikong/$name_for_run/production_run/${rsdir}
mkdir -p $outdir


#srun -N 4 -n 16 -c 16 shifter  --module=mpich-cle6 --image=driftingpig/obiwan_scarlet python mpi_scarlet.py
shifter  --image=driftingpig/obiwan_scarlet --volume=$HOME":/homedir" /bin/bash
