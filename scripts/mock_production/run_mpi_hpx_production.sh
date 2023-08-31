#!/bin/sh

if [ "$#" -lt 2 ]
then
echo "Run mpi processing of healpix file list (request ranks = total number)"
echo "Usage: run_mpi_hpx_production hpx_list (0-17, image, test#) z_range (0-2)"
echo "       filename for healpix list will be pixels_${1}.txt"
echo "       optional 3rd parameter: name of yaml config file to use"
echo "       default (diffsky_config) (.yaml is assumed)"
exit
else
hpx_list="${1}"
z_range=${2}
echo "Running from `pwd`"
pixels_list="pixels_${hpx_list}.txt"
echo "pixels_list=${pixels_list}"
echo "z_range=${z_range}"
if [ "$#" -gt 2 ]
then
config_file=${3}
echo "config_file=${config_file}"
xtra_args="-config_file ${config_file}"
else
xtra_args=""
fi
fi

source activate diffsky
PYTHONPATH=/home/ekovacs/.conda/envs/diffsky/bin/python
export PYTHONPATH

#retrive number of pixels in list file
total_pix_num="`wc -l < ${pixels_list}`"
echo "total_pix_num=${total_pix_num}"

script_name=run_diffsky_healpix_production.py
pythonpath=/home/ekovacs/.conda/envs/diffsky/bin/python
args="${pixels_list} -zrange_value ${z_range} ${xtra_args}"

mpiexec -n ${total_pix_num} ${pythonpath} ${script_name} ${args}
echo "Running ${pythonpath} ${script_name} ${args} on ${total_pix_num} ranks"
