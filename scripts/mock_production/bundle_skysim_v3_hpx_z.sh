#!/bin/sh

if [ "$#" -lt 2 ]
then
echo "Bundle jobs to selected nodes (4 pixels per node)"
echo "Usage: bundle_skysim_v3_hpx_z hpx_group (0-11, image) z_range (0-2)"
exit
else
hpx_group=${1}
z_range=${2}
echo "hpx_group=${hpx_group}"
echo "z_range=${z_range}"
fi

NODES=`cat $COBALT_NODEFILE | wc -l`
PROCS=1
npix=1
# starting pixel number (1 is start of file)
#cd /home/ekovacs/cosmology/mock_production/skysim5000_v3/OR_5000_v2
cd /lus/eagle/projects/LastJourney/kovacs/Catalog_5000/OR_5000/skysim_v3.1.0_production
echo "Running from `pwd`"
source activate diffmah
PYTHONPATH=/home/ekovacs/.conda/envs/diffmah/bin/python
export PYTHONPATH

vprod="skysim_v3.1.0_production"
xtra_args="-skip_synthetics"
filename="cutout"
tot_pix_grp=16
if [ "$hpx_group" == "image" ]
then
#131 pixels
total_pix_num=132
else
if [ "$hpx_group" -lt "$tot_pix_grp" ]
then
# 128 pixels per file
total_pix_num=129
else
# 74 pixels per file
total_pix_num=75
fi
fi
echo "total_pix_num=${total_pix_num}"

script_name=run_skysim_healpix_production.py
pythonpath=/home/ekovacs/.conda/envs/diffmah/bin/python

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
  for pixnumber in {1..3}
  do
  if [ "$npix" -lt "$total_pix_num" ]
  then
  pixelname=$(cat pixels_${hpx_group}.txt | awk 'NR=='${npix})
  echo $pixelname
  echo "${pixelname}_${z_range}" >> started_pixels_${hpx_group}_${z_range}.txt
  filename2=${filename}_${pixelname}.hdf5
  args="${filename2} -zrange_value ${z_range} ${xtra_args}"
  #echo $args
  #   mpirun --host ${hostname1}
  #echo ${hostname1}_${COBALT_JOBID}_${pixelname}-err.log
  jobname="cutout_${pixelname}_z_${z_range}_${vprod}_${hostname1}_${COBALT_JOBID}.log"
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
