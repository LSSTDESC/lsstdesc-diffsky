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
hpx_list="pixels_${1}.txt"
z_range=${2}
vprod="diffsky_v0.1.0_production"
cd /lus/eagle/projects/LastJourney/kovacs/Catalog_5000/OR_5000/${vprod}
echo "Running from `pwd`"

echo "hpx_list=${hpx_list}"
echo "z_range=${z_range}"
if [ "$#" -gt 2 ]
then
config_file=${3}
echo "config_file=${config_file}"
xtra_args="-config_file ${config_file}"
cname=${config_file}
else
xtra_args=""
cname=${vprod}
fi
fi

source activate diffsky
PYTHONPATH=/home/ekovacs/.conda/envs/diffsky/bin/python
export PYTHONPATH

tot_pix_grp=17
if [ "$hpx_group" == "image" ]
then
#131 pixels
total_pix_num=131
else
if [[ "${1}" =~ "test" ]]; then
#test# gives number of pixels
total_pix_num=$(expr "${hpx_group#*test}")
else
if [ "$hpx_group" -lt "$tot_pix_grp" ]
then
# 128 pixels per file
total_pix_num=128
else
# 74 pixels in last file
total_pix_num=74
fi
fi
fi
echo "total_pix_num=${total_pix_num}"

script_name=run_diffsky_healpix_production.py
pythonpath=/home/ekovacs/.conda/envs/diffsky/bin/python
args="${hpx_list} -zrange_value ${z_range} ${xtra_args}"

mpiexec -n ${total_pix_num} ${pythonpath} ${script_name} ${args}
