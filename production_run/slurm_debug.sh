#!/bin/bash -l

#SBATCH -p debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --account=desi
#SBATCH -J t967
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

export LEGACYVER=DR9.6.7
export rowstart=0
export name_for_run=decals_ngc
export dataset=normal
export nobj=200
export threads=5
#decam for south, north for north
export run_region=decam
#only need to be set while running cosmos 
#export cosmos_section=$1  
#export rsdir=rs${rowstart}_cosmos${cosmos_section}
export rsdir=more_rs${rowstart}

export PYTHONPATH=./mpi:$PYTHONPATH

export topdir=$CSCRATCH/Obiwan/obiwan_testrun/
export outdir=$topdir/$name_for_run/production_run/${rsdir}
mkdir -p $outdir

#1. If you just want to create and save plots without viewing them, you can turn off interactive mode in matplotlib using matplotlib.pyplot.ioff() function. See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ioff.html
#2. Otherwise, if you need the interactive display, you will need to volume mount your $HOME/.Xauthority file into your container when you launch Shifter. You can read more about volume mounting in Shifter here. https://docs.nersc.gov/development/shifter/how-to-use/#volume-mounting-in-shifter.
#3. plots are only for debug purpose, there are many places that you can turn on plotting in legacypipe. In production runs, this is not needed: #1/#2 are bothnot required. 
shifter --module=mpich-cle6 --volume=$HOME":/homedir" --image=legacysurvey/legacypipe:DR9.6.7 ./run_debug.sh 1561p017

#to enter this shifter environment on terminal:
#shifter --module=mpich-cle6 --volume=$HOME":/homedir" --image=legacysurvey/legacypipe:DR9.6.7 /bin/bash
