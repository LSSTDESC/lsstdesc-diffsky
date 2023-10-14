#!/bin/sh

if [ "$#" -lt 2 ]
then
echo "Run mpi processing of healpix file list (request ranks = total number)"
echo "Usage: run_mpi_hpx_production hpx_list (0-17, image, test#) z_range (0-2)"
echo "       filename for healpix list will be pixels_${1}.txt"
echo "       optional 3rd parameter: name of yaml config file to use"
echo "           default (diffsky_config) (.yaml is assumed)"
echo "           dirname storing config file is supplied_config_filename_production"
echo "           default (production)" 
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
production_dir="${config_file}_production"
echo "production_dir=${production_dir}"
xtra_args="-config_file ${config_file} -production_dir ${production_dir}"
echo "xtra_args=${xtra_args}"
else
xtra_args=""
fi
fi

source activate diffsky_v3

#retrive number of pixels in list file
total_pix_num="`wc -l < ${pixels_list}`"
echo "total_pix_num=${total_pix_num}"

script_name=run_diffsky_healpix_production.py
pythonpath=/home/ekovacs/.conda/envs/diffsky_v3/bin/python
args="${pixels_list} -zrange_value ${z_range} ${xtra_args}"

mpiexec -n ${total_pix_num} ${pythonpath} ${script_name} ${args}
echo "Running ${pythonpath} ${script_name} ${args} on ${total_pix_num} ranks"
