#!/bin/sh

if [ "$#" -lt 2 ]
then
echo "Bundle jobs to selected nodes (4 pixels per node)"
echo "Usage: bundle_diffsky_hpx_z hpx_group (0-17, image, test) z_range (0-2)"
echo "       optional 3rd parameter: name of yaml config file to use"
echo "       default (diffsky_config) (.yaml is assumed)"
exit
else
hpx_group=${1}
z_range=${2}
vprod="diffsky_v0.1.0_production"
cd /lus/eagle/projects/LastJourney/kovacs/Catalog_5000/OR_5000/${vprod}
echo "Running from `pwd`"

echo "hpx_group=${hpx_group}"
echo "z_range=${z_range}"
echo "num=${hpx_group#*test}"
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

NODES=`cat $COBALT_NODEFILE | wc -l`
PROCS=1
npix=1
# starting pixel number (1 is start of file)
source activate diffsky
PYTHONPATH=/home/ekovacs/.conda/envs/diffsky/bin/python
export PYTHONPATH

tot_pix_grp=16
if [ "$hpx_group" == "image" ]
then
#131 pixels
total_pix_num=132
else
if [[ "${1}" =~ "test" ]]; then
#test# gives number of pixels
total_pix_num=$(expr "${hpx_group#*test}" + 1)
else
if [ "$hpx_group" -lt "$tot_pix_grp" ]
then
# 128 pixels per file
total_pix_num=129
else
# 74 pixels in last file
total_pix_num=75
fi
fi
fi
echo "total_pix_num=${total_pix_num}"

script_name=run_diffsky_healpix_production.py
pythonpath=/home/ekovacs/.conda/envs/diffsky/bin/python

readarray nodenumbers < $COBALT_NODEFILE
for nodenumber in "${nodenumbers[@]}"
#for nodenumber in {1..3}
do
  hostname1=$nodenumber
  #hostname1=expr xargs $nodenumber
  #hostname1=${ xargs $nodenumber}
  hostname1=${hostname1%?}
  #echo $hostname1
  #hostname1=$(cat $COBALT_NODEFILE | awk 'NR=='${nodenumber})
  for pixnumber in {1..4}
  do
  if [ "$npix" -lt "$total_pix_num" ]
  then
  pixelname=$(cat pixels_${hpx_group}.txt | awk 'NR=='${npix})
  echo $pixelname
  echo "${pixelname}_${z_range}" >> started_pixels_${hpx_group}_${z_range}.txt
  args="${pixelname} -zrange_value ${z_range} ${xtra_args}"
  #echo $args
  #   mpirun --host ${hostname1}
  #echo ${hostname1}_${COBALT_JOBID}_${pixelname}-err.log
  jobname="logfiles/cutout_${pixelname}_z_${z_range}_${cname}_${hostname1}_${COBALT_JOBID}.log"
  mpirun --host ${hostname1} -n $PROCS $pythonpath $script_name ${args} > $jobname 2>&1 &
  #mpirun --host ${hostname1} -n $PROCS $pythonpath $script_name ${args} > "${jobname}_1.log" 2>&1 
  npix=$(expr $npix + 1 )
  else
  echo "$npix > maximum number of pixels"
  fi
  done
done
echo "$npix $z_range" >> restart_npix_${hpx_group}_${z_range}.txt
wait
