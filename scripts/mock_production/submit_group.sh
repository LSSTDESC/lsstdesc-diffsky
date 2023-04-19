#!/bin/sh
export EMAIL=kovacs@anl.gov

if [ "$#" -lt 1 ]
then
echo "Submit jobs for healpix group for all z ranges"
echo "Usage: submit_group hpx_group (0-11)"
echo "     : special case for cosmodc2 area"
echo "     :    submit_group image"
exit
else
hpx_group=${1}
echo "hpx_group=${hpx_group}"
fi

tot_pix_grp=12
if [ "$hpx_group" == "image" ]
then
# 131 pixels in file; 8 pixels per node
nodes=17
else
if [ "$hpx_group" -lt "$tot_pix_grp" ]
then
# 128 pixels per file; 8 pixels per node
nodes=16
else
if [ "$hpx_group" -gt "$tot_pix_grp" ]
then
#special case for reruns
nodes=1
else
# 32 pixels per file; 8 pixels per node                   
nodes=4
fi
fi
fi

qsub -n ${nodes} -t 11:00:00 -A LastJourney -M ${EMAIL} ./bundle_skysim_v3_hpx_z.sh ${hpx_group} 0
qsub -n ${nodes} -t 11:00:00 -A LastJourney -M ${EMAIL} ./bundle_skysim_v3_hpx_z.sh ${hpx_group} 1
qsub -n ${nodes} -t 11:00:00 -A LastJourneu -M ${EMAIL} ./bundle_skysim_v3_hpx_z.sh ${hpx_group} 2
