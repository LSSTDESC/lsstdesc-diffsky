"""
Production pipeline main driver script to produce mock galaxy catalog
from HACC simulation data products

Sample command to run script on halo lightcone healpixel:
python run_diffsky_healpix_production.py cutout_9554.hdf5 -zrange_value 1 -skip_synthetics\
    |& tee logfiles/cutout_9554_z_1_2_smdpl_replace_nofit_alt_dust.log

"""
import sys
import os
import glob
import argparse
import yaml
import numpy as np
from os.path import expanduser
import subprocess
from lsstdesc_diffsky.halo_information.get_healpix_cutout_info import get_healpix_cutout_info
from lsstdesc_diffsky.write_mock_to_disk import write_umachine_healpix_mock_to_disk


def retrieve_commit_hash(path_to_repo):
    """ Return the commit hash of the git branch currently live in the input path.
    Parameters
    ----------
    path_to_repo : string
    Returns
    -------
    commit_hash : string
    """
    cmd = 'cd {0} && git rev-parse HEAD'.format(path_to_repo)
    return subprocess.check_output(cmd, shell=True).strip()


home = expanduser("~")
path_to_lsstdesc_diffsky = os.path.join(home, 'cosmology/lsstdesc-diffsky')
sys.path.insert(0, path_to_lsstdesc_diffsky)

# pre-import jax modules to get around import error


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("healpix_number",
                    help="Healpix cutout number to run")
# parser.add_argument("commit_hash",
#    help="Commit hash to save in output files")
parser.add_argument("-input_master_dirname",
                    help="Directory name (relative to home) storing sub-directories of input files",
                    default='Catalog_5000/OR_5000')
parser.add_argument("-config_dirname",
                    help="Directory name storing link to production config yaml file",
                    default='production')
parser.add_argument("-config_filename",
                    help="filename for production config yaml file (.yaml assumed)",
                    default='diffsky_config')
parser.add_argument("-zrange_value",
                    help="z-range to run",
                    choices=['0', '1', '2', 'all'],
                    default='all')
parser.add_argument("-verbose",
                    help="Turn on extra printing",
                    action='store_true', default=False)
parser.add_argument("-ndebug_snaps",
                    help="Number of debug snapshots to save",
                    type=int, default=-1)
args = parser.parse_args()

# setup directory names; read yaml configuration
input_master_dirname = os.path.join(home, args.input_master_dirname)
yaml_dir = os.path.join(input_master_dirname, args.config_dirname)
yaml_fn = os.path.join(yaml_dir, args.config_filename + '.yaml')
with open(os.path.join(yaml_dir, yaml_fn), 'r') as fh:
    inputs = yaml.safe_load(fh)

versionMajor = inputs['version']['versionMajor']
versionMinor = inputs['version']['versionMinor']
versionMinorMinor = inputs['version']['versionMinorMinor']

name = inputs['name']

healpix_cutout_dirname = os.path.join(input_master_dirname,
                                      inputs['file_dirs']['healpix_cutout_dirname'])
output_mock_dirname = os.path.join(
    input_master_dirname,
    inputs['file_dirs']['output_mock_dirname'].format(versionMajor,
                                                      versionMinor, versionMinorMinor),
    name)
shape_dir = os.path.join(input_master_dirname, inputs['file_dirs']['shape_dirname'])
pkldirname = os.path.join(input_master_dirname, inputs['file_dirs']['pkldirname'])
healpix_fname_template = inputs['file_names']['healpix_fname']

SED_pars = {}

print('Setting master directory to {}'.format(input_master_dirname))
print('Reading inputs from {}'.format(healpix_cutout_dirname))
print('Writing outputs to {}'.format(output_mock_dirname))

commit_hash = retrieve_commit_hash(path_to_lsstdesc_diffsky)[0:7]
print('Using commit hash {}'.format(commit_hash))

if args.verbose:
    print("paths=", home, path_to_lsstdesc_diffsky, sys.path)

# loop over z-ranges
if args.zrange_value == 'all':
    z_range_dirs = [
        os.path.basename(d) for d in glob.glob(
            healpix_cutout_dirname +
            '/*') if 'z' in d]
else:
    z_range_dirs = [os.path.basename(d) for d in glob.glob(
        healpix_cutout_dirname + '/z_{}*'.format(args.zrange_value))]

# setup environmentals to limit pthreads
for env in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[env] = "1"
    print("Setting {} = {}".format(env, os.environ.get(env)))

for zdir in z_range_dirs:
                         
    # get list of snapshots
    healpix_cutout_fname = os.path.join(
        healpix_cutout_dirname, zdir, healpix_fname_template.format(args.healpix_number))
    print('Processing healpix cutout {}'.format(healpix_cutout_fname))
    healpix_data, redshift_strings, snapshots, z2ts = get_healpix_cutout_info(pkldirname,
                                                                              healpix_cutout_fname, sim_name='AlphaQ')

    if args.ndebug_snaps > 0:

        redshift_strings = redshift_strings[-args.ndebug_snaps:]
        snapshots = snapshots[-args.ndebug_snaps:]
    expansion_factors = [1. / (1 + float(z)) for z in redshift_strings]
    if args.verbose:
        print("target z's and a's:", redshift_strings, expansion_factors)

    if len(snapshots) > 0:
        umachine_mstar_ssfr_mock_dirname = (
            os.path.join(input_master_dirname, inputs['file_dirs']['um_sfr_catalogs_dirname']))
        sfr_files = sorted([os.path.basename(f) for f in glob.glob(
            umachine_mstar_ssfr_mock_dirname + '/sfr*.h*')])
        um_expansion_factors = np.asarray(
            [float(f.split('sfr_catalog_')[-1].split(inputs['parse_str']['um_split_fname'])[0]) for f in sfr_files])
        closest_snapshots = [np.abs(um_expansion_factors - a).argmin()
                             for a in expansion_factors]
        if (args.verbose):
            print('index of closest snapshots:', closest_snapshots)

        umachine_mstar_ssfr_mock_basename_list = [
            sfr_files[n] for n in closest_snapshots]
        umachine_mstar_ssfr_mock_fname_list = list(
            (os.path.join(umachine_mstar_ssfr_mock_dirname, basename)
             for basename in umachine_mstar_ssfr_mock_basename_list))
        if (args.verbose):
            print('umachine_mstar_ssfr_mock_basename_list:',
                  umachine_mstar_ssfr_mock_basename_list)

        healpix_basename = os.path.basename(args.healpix_fname)
        output_mock_basename = '_'.join(
            [inputs['file_dirs']['output_mock_dirname'].split('_')[0], zdir, healpix_basename])
        output_healpix_mock_fname = os.path.join(
            output_mock_dirname, output_mock_basename)
        # if(args.verbose):
        print('Writing output_healpix_mock_fname: {}\n'.format(output_healpix_mock_fname))

        redshift_list = [float(z) for z in redshift_strings]

        for k, v in inputs.items():
            if k in ['dust_parameters', 'SEDs', 'empirical_models']:
                for par, vv in v.items():
                    if 'dirname' in par:
                        SED_pars[par] = os.path.join(input_master_dirname, vv)
                    else:
                        SED_pars[par] = vv
                    print('Assigning {} to {}'.format(par, SED_pars[par]))

        cosmological_params = {'H0': inputs['cosmology']['H0'],
                               'OmegaM': inputs['cosmology']['OmegaM'],
                               'OmegaB': inputs['cosmology']['OmegaB'],
                               'w0': inputs['cosmology']['w0'],
                               'wa': inputs['cosmology']['wa'],
                              }
        print()

        write_umachine_healpix_mock_to_disk(
            umachine_mstar_ssfr_mock_fname_list,
            healpix_data, snapshots, output_healpix_mock_fname, shape_dir,
            redshift_list, commit_hash,
            versionMajor=versionMajor, versionMinor=versionMinor,
            versionMinorMinor=versionMinorMinor,
            SED_pars=SED_pars,
            synthetic_params=inputs['synthetic_ultra_faints'],
            shear_params=inputs['shears'],
            Nside=inputs['nside'], z2ts=z2ts, cosmological_params=cosmological_params,
            )
    else:
        print('Skipping empty healpix-cutout file {}'.format(args.healpix_fname))
