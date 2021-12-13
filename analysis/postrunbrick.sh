#!/bin/bash

source /project/projectdirs/desi/software/desi_environment.sh 20.8

for VARIABLE in {0..29}
do
	echo running $VARIABLE of 30
	srun -N 1 -n 1 -c 64 python postrunbrick.py --survey-dir /global/project/projectdirs/cosmo/data/legacysurvey/dr9/ --outdir /global/cfs/cdirs/desi/users/huikong/decals_ngc/production_run/ --threads 30 --dataset normal --subsection more_rs0 --nobj 200 --startid 0 --tot-slice 30 --slice-idx $VARIABLE &
done

wait
