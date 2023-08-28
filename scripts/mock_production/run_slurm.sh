#!/bin/bash
#
#SBATCH -J diffsky_array
#SBATCH -A halotools
#SBATCH -p bdw
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --array=1-4

ARGS=(9425 9426 9554 9681)
source activate diffsky
srun --cpu-bind=cores python run_diffsky_healpix_production.py ${ARGS[$SLURM_ARRAY_TASK_ID]} -zrange_value 0 -config_filename diffsky_v0.1.0_pscan_3765 > \
logfiles/cutout_${ARGS[$SLURM_ARRAY_TASK_ID]}_z_0_pscan_3765_slurm_${SLURM_JOB_ID}.log
