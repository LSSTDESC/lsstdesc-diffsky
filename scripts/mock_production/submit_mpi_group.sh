#!/bin/sh
export EMAIL=kovacs@anl.gov

if [ "$#" -lt 2 ]
then
echo "Submit jobs for healpix group for all z ranges"
echo "Usage: submit_mpi_group hpx_group (0-17) yaml_config_file"
echo "     : special case for cosmodc2 area"
echo "     :    submit_group image"
echo "     : special case for test area"
echo "     :    submit_group test*"
echo "     : 2nd parameter specifies yaml config file"
echo "     : optional 3rd parameter specifies number of nodes"
exit
else
hpx_group=${1}
echo "hpx_group=${hpx_group}"
config_file=${2}
echo "config_file=${config_file}"
fi

# setup default nodes
tot_pix_grp=16
if [ "$hpx_group" == "image" ]
then
# 131 pixels in file; 4 pixels per node
nodes=33
else
if [[ "${hpx_group}" =~ "test" ]]
then
nodes=6
else
if [ "$hpx_group" -lt "$tot_pix_grp" ]
then
# 128 pixels per file; 4 pixels per node
nodes=32
else
if [ "$hpx_group" -gt "$tot_pix_grp" ]
then
#special case for reruns
nodes=1
else
# special end case: 32 pixels per file; 4 pixels per node                   
nodes=8
fi
fi
fi
fi
# check for requested nodes
if [ "$#" -gt 2 ]
then
nodes=${3}
fi
echo "nodes=${nodes}"

qsub -n ${nodes} -t 4:00:00 -A LastJourney -M ${EMAIL} --attrs filesystems=home,eagle ./run_mpi_hpx_production.sh ${hpx_group} 0 ${config_file}
qsub -n ${nodes} -t 4:00:00 -A LastJourney -M ${EMAIL} --attrs filesystems=home,eagle ./run_mpi_hpx_production.sh ${hpx_group} 1 ${config_file}
qsub -n ${nodes} -t 4:00:00 -A LastJourney -M ${EMAIL} --attrs filesystems=home,eagle ./run_mpi_hpx_production.sh ${hpx_group} 2 ${config_file}
