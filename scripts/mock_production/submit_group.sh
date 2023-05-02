#!/bin/sh
export EMAIL=kovacs@anl.gov

if [ "$#" -lt 1 ]
then
echo "Submit jobs for healpix group for all z ranges"
echo "Usage: submit_group hpx_group (0-11)"
echo "     : special case for cosmodc2 area"
echo "     :    submit_group image"
echo "     : special case for test area"
echo "     :    submit_group test"
exit
else
hpx_group=${1}
echo "hpx_group=${hpx_group}"
fi

tot_pix_grp=12
if [ "$hpx_group" == "image" ]
then
# 131 pixels in file; 4 pixels per node
nodes=33
else
if [ "$hpx_group" == "test" ]
then
nodes=1
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

qsub -n ${nodes} -t 11:00:00 -A LastJourney -M ${EMAIL} ./bundle_diffsky_hpx_z.sh ${hpx_group} 0
qsub -n ${nodes} -t 11:00:00 -A LastJourney -M ${EMAIL} ./bundle_diffsky_hpx_z.sh ${hpx_group} 1
qsub -n ${nodes} -t 11:00:00 -A LastJourneu -M ${EMAIL} ./bundle_diffsky_hpx_z.sh ${hpx_group} 2
