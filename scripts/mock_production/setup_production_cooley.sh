#!/bin/sh
if [ "$#" -lt 1 ]
then
echo "Supply version number eg. 0.3.1"
exit
fi

basename="roman_rubin_2023"
pix_list="test_elais"
catname=${basename}_v${1}
mkdir ${catname}
mkdir ${catname}_production

cd ${catname}_production
ln -s /home/ekovacs/cosmology/lsstdesc-diffsky/lsstdesc_diffsky/AlphaQ_z2ts.pkl
ln -s /home/ekovacs/cosmology/lsstdesc-diffsky/scripts/mock_production/pixels_${pix_list}.txt
ln -s /home/ekovacs/cosmology/lsstdesc-diffsky/scripts/mock_production/run_diffsky_healpix_production.py
ln -s /home/ekovacs/cosmology/lsstdesc-diffsky/scripts/mock_production/run_mpi_hpx_production.sh
ln -s /home/ekovacs/cosmology/lsstdesc-diffsky/scripts/mock_production/submit_mpi_group.sh
ln -s /home/ekovacs/cosmology/lsstdesc-diffsky/lsstdesc_diffsky/config_files/${catname}.yaml
mkdir cobalt
mkdir logfiles

echo "`pwd`"
ls -l *
