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

parser.add_argument("healpix_fname",
                    help="Filename of healpix cutout to run")
# parser.add_argument("commit_hash",
#    help="Commit hash to save in output files")
parser.add_argument("-input_master_dirname",
                    help="Directory name (relative to home) storing sub-directories of input files",
                    default='Catalog_5000/OR_5000')
parser.add_argument("-healpix_cutout_dirname",
                    help="Directory name (relative to input_master_dirname) storing healpix cutout files",
                    default='healpix_cutouts')
parser.add_argument("-config_dirname",
                    help="Directory name storing link to production config yaml file",
                    default='production')
parser.add_argument("-config_filename",
                    help="filename for production config yaml file",
                    default='diffsky_config.yaml')
parser.add_argument("-um_sfr_catalogs_dirname",
                    help="Directory name (relative to input_master_dirname) storing um input sfr catalogs",
                    default='smdpl_value_added_replaced_nofit_catalogs')
parser.add_argument("-um_parse_filename",
                    help="String on which to split um sfr catalog filename to extract scale parameter",
                    default='.diffstar_fits')
parser.add_argument("-output_mock_dirname",
                    help="Directory name (relative to input_master_dirname) storing output mock healpix files",
                    default='diffsky_v{}.{}.{}')
parser.add_argument("-shape_dirname",
                    help="Directory name (relative to input_master_dirname) storing halo shape files",
                    default='OR_haloshapes')
parser.add_argument("-dsps_data_dirname",
                    help="Directory name (relative to input_master_dirname) storing dsps data",
                    default='dsps_data/FSPS_ssp_data')
parser.add_argument("-pkldirname",
                    help="Directory name (relative to home) storing pkl file with snapshot <-> redshift correspondence",
                    default='cosmology/lsstdesc-diffsky/lsstdesc_diffsky')
parser.add_argument("-zrange_value",
                    help="z-range to run",
                    choices=['0', '1', '2', 'all'],
                    default='all')
parser.add_argument("-synthetic_mass_min",
                    help="Value of minimum halo mass for synthetic halos",
                    type=float, default=9.8)
parser.add_argument("-use_satellites",
                    help="Use satellite synthetic low-mass galaxies",
                    action='store_true', default=False)
parser.add_argument("-verbose",
                    help="Turn on extra printing",
                    action='store_true', default=False)
parser.add_argument("-skip_synthetics",
                    help="Skip low mass synthetic galaxies",
                    action='store_true', default=False)
parser.add_argument("-dust",
                    help="Include dust attenuation",
                    action='store_false', default=True)
parser.add_argument("-use_diffmah_pop",
                    help="Use diffmah_pop for resampled 'no-fit' galaxies",
                    action='store_false', default=True)
parser.add_argument("-nside",
                    help="Nside used to create healpixels",
                    type=int, default=32)
parser.add_argument("-dz",
                    help="Value of dz interval for SSP interpolation",
                    type=float, default=0.02)
parser.add_argument("-ndebug_snaps",
                    help="Number of debug snapshots to save",
                    type=int, default=-1)
parser.add_argument("-H0",
                    help="Hubble constant",
                    type=float, default=71.0)
parser.add_argument("-OmegaM",
                    help="OmegaM",
                    type=float, default=0.2648)
parser.add_argument("-OmegaB",
                    help="OmegaB",
                    type=float, default=0.0448)
parser.add_argument("-w0",
                    help="w0",
                    type=float, default=-1.0)
parser.add_argument("-wa",
                    help="wa",
                    type=float, default=0.0)
parser.add_argument("-versionMajor",
                    help="Major version number",
                    type=int, default=0)
parser.add_argument("-versionMinor",
                    help="Minor version number",
                    type=int, default=1)
parser.add_argument("-versionMinorMinor",
                    help="MinorMinor version number",
                    type=int, default=0)
args = parser.parse_args()

# setup directory names; read yaml configuration
input_master_dirname = os.path.join(home, args.input_master_dirname)
yaml_dir = os.path.join(input_master_dirname, args.config_dirname)
yaml_fn = os.path.join(yaml_dir, args.config_filename.format(args.versionMajor, args.versionMinor,
                                                                   args.versionMinorMinor))
inputs = yaml.safe_load(yaml_fn)
print(inputs)

pkldirname = os.path.join(home, args.pkldirname)
healpix_cutout_dirname = os.path.join(input_master_dirname, args.healpix_cutout_dirname)
output_mock_dirname = os.path.join(input_master_dirname,
                                   args.output_mock_dirname.format(args.versionMajor, args.versionMinor,
                                                                   args.versionMinorMinor))
shape_dir = os.path.join(input_master_dirname, args.shape_dirname)


print('Setting master directory to {}'.format(input_master_dirname))
print('Reading inputs from {}'.format(healpix_cutout_dirname))
print('Writing outputs to {}'.format(output_mock_dirname))

commit_hash = retrieve_commit_hash(path_to_lsstdesc_diffsky)[0:7]
print('Using commit hash {}'.format(commit_hash))
synthetic_halo_minimum_mass = args.synthetic_mass_min
use_centrals = not (args.use_satellites)
SED_pars = {}
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


for zdir in z_range_dirs:

    # get list of snapshots
    healpix_cutout_fname = os.path.join(
        healpix_cutout_dirname, zdir, args.healpix_fname)
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
            os.path.join(input_master_dirname, args.um_sfr_catalogs_dirname))
        sfr_files = sorted([os.path.basename(f) for f in glob.glob(
            umachine_mstar_ssfr_mock_dirname + '/sfr*.h*')])
        um_expansion_factors = np.asarray(
            [float(f.split('sfr_catalog_')[-1].split(args.um_parse_filename)[0]) for f in sfr_files])
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
            [args.output_mock_dirname.split('_')[0], zdir, healpix_basename])
        output_healpix_mock_fname = os.path.join(
            output_mock_dirname, output_mock_basename)
        # if(args.verbose):
        print('Writing output_healpix_mock_fname: {}\n'.format(output_healpix_mock_fname))

        redshift_list = [float(z) for z in redshift_strings]

        argsdict = vars(args)
        for par in ['dsps_data_dirname', 'dust']:
            if 'dirname' in par :
                SED_pars[par] = os.path.join(input_master_dirname, argsdict[par])
            else:
                SED_pars[par] = argsdict[par]
            print('Assigning {} to {}'.format(par, SED_pars[par]))

        cosmological_params = {'H0': args.H0,
                               'OmegaM': args.OmegaM,
                               'OmegaB': args.OmegaB,
                               'w0': args.w0,
                               'wa': args.wa,
                              }

        write_umachine_healpix_mock_to_disk(
            umachine_mstar_ssfr_mock_fname_list,
            healpix_data, snapshots, output_healpix_mock_fname, shape_dir,
            redshift_list, commit_hash, skip_synthetics=args.skip_synthetics,
            dz=args.dz, versionMajor=args.versionMajor, versionMinor=args.versionMinor,
            versionMinorMinor=args.versionMinorMinor, SED_pars=SED_pars,
            synthetic_halo_minimum_mass=synthetic_halo_minimum_mass,
            use_centrals=use_centrals,
            Nside=args.nside, z2ts=z2ts, cosmological_params=cosmological_params,
            use_diffmah_pop=args.use_diffmah_pop,
            )
    else:
        print('Skipping empty healpix-cutout file {}'.format(args.healpix_fname))