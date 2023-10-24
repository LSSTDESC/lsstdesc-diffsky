"""
write_mock_to_disk.py
=====================
Main module for production of mock galaxy catalogs for LSST DESC.
"""
import gc
import os
import re
import threading
import warnings
from time import time

import h5py
import numpy as np
import psutil
from astropy.cosmology import WMAP7, FlatLambdaCDM
from astropy.table import Table, vstack
from astropy.utils.misc import NumpyRNGContext
from diffstar.defaults import FB
from diffstar.sfh import sfh_galpop
from dsps.metallicity.mzr import DEFAULT_MET_PDICT
from galsampler import crossmatch
from galsampler.galmatch import galsample
from halotools.empirical_models import halo_mass_to_halo_radius

# Galsampler
from halotools.utils.value_added_halo_table_functions import compute_uber_hostid
from halotools.utils.vector_utilities import normalized_vectors
from jax import random as jran

from .black_hole_modeling.black_hole_accretion_rate import monte_carlo_bh_acc_rate
from .black_hole_modeling.black_hole_mass import monte_carlo_black_hole_mass
from .defaults import CosmoParams
from .diffstarpop.mc_diffstar import mc_diffstarpop
from .dspspop.boris_dust import DEFAULT_U_PARAMS as DEFAULT_BORIS_U_PARAMS
from .dspspop.burstshapepop import DEFAULT_BURSTSHAPE_U_PARAMS
from .dspspop.dust_deltapop import DEFAULT_DUST_DELTA_U_PARAMS
from .dspspop.lgavpop import DEFAULT_LGAV_U_PARAMS
from .dspspop.lgfburstpop import DEFAULT_LGFBURST_U_PARAMS
from .ellipticity_modeling.ellipticity_model import monte_carlo_ellipticity_bulge_disk

# Halo shapes
from .halo_information.get_fof_halo_shapes import get_halo_shapes, get_matched_shapes

# Synthetics
from .halo_information.get_healpix_cutout_info import get_snap_redshift_min

# SED generation
from .legacy.roman_rubin_2023.dsps.data_loaders.load_ssp_data import (
    load_ssp_templates_singlemet,
)
from .param_data.param_reader import DiffskyPopParams
from .pecZ import pecZ
from .photometry.get_SFH_from_params import (
    get_diff_params,
    get_log_safe_ssfr,
    get_logsm_sfr_obs,
)
from .photometry.load_filter_data import assemble_filter_data, get_filter_wave_trans
from .photometry.photometry_lc_interp_singlemet import get_diffsky_sed_info_singlemet
from .photometry.precompute_ssp_tables import (
    precompute_ssp_obsmags_on_z_table_singlemet,
    precompute_ssp_restmags_singlemet,
)

# Additional catalog properties
from .size_modeling.zhang_yang17 import (
    mc_size_vs_luminosity_early_type,
    mc_size_vs_luminosity_late_type,
)
from .synthetic_subhalos.extend_subhalo_mpeak_range import (
    create_synthetic_lowmass_mock_with_centrals,
    map_mstar_onto_lowmass_extension,
)
from .synthetic_subhalos.synthetic_cluster_satellites import (
    model_synthetic_cluster_satellites,
)
from .synthetic_subhalos.synthetic_lowmass_subhalos import synthetic_logmpeak
from .triaxial_satellite_distributions.axis_ratio_model import monte_carlo_halo_shapes
from .triaxial_satellite_distributions.monte_carlo_triaxial_profile import (
    generate_triaxial_satellite_distribution,
)

# metadata
from .infer_diffcode_versions import infer_software_versions

fof_halo_mass = "fof_halo_mass"
# fof halo mass in healpix cutouts
fof_mass = "fof_mass"
mass = "mass"
fof_max = 14.5
sod_mass = "sod_mass"
m_particle_1000 = 1.85e12

Nside_sky = 2048  # fine pixelization for determining sky area

# halo id offsets
# offset to generate unique id for cutouts and snapshots
cutout_id_offset_halo = int(1e3)
# offset to guarantee unique halo ids across cutout files and snapshots
halo_id_offset = int(1e8)

# galaxy id offsets for non image-sim catalogs (eg. 5000 sq. deg.)
cutout_id_offset = int(1e9)
z_offsets_not_im = {"32": [0, 1e8, 2e8, 3e8]}

# constants to determine synthetic number density
Ntotal_synthetics = 1932058570  # total number of synthetic galaxies in cosmoDC2_image
nhpx_total = float(131)  # number of healpixels in image area
snapshot_min = 121
# specify edges of octant
volume_minx = 0.0
volume_miny = 0.0
volume_maxz = 0.0

__all__ = (
    "write_umachine_healpix_mock_to_disk",
    "build_output_snapshot_mock",
    "write_output_mock_to_disk",
)

warnings.filterwarnings("ignore", category=DeprecationWarning)


def write_umachine_healpix_mock_to_disk(
    umachine_mstar_ssfr_mock_fname_list,
    healpix_data,
    snapshots,
    output_color_mock_fname,
    shape_dir,
    redshift_list,
    commit_hash,
    SED_pars={},
    cosmological_params={},
    synthetic_params={},
    shear_params={},
    versionMajor=0,
    versionMinor=1,
    versionMinorMinor=0,
    Nside=32,
    mstar_min=7e6,
    z2ts={},
    Lbox=3000.0,
    num_synthetic_gal_ratio=1.0,
    mass_match_noise=0.1,
):
    """
    GalSample the UM mock into the lightcone healpix cutout,
    compute the SEDs using DSPS and
    write the healpix mock to disk.

    Parameters
    ----------
    umachine_mstar_ssfr_mock_fname_list : list
        List of length num_snaps storing the absolute path to the
        value-added UniverseMachine snapshot mock

    healpix_data : <HDF5 file>
        Pointer to open hdf5 file for the lightcone healpix cutout
        source halos into which UniverseMachine will be GalSampled

    snapshots : list
        List of snapshots in lightcone healpix cutout

    output_color_mock_fname : string
        Absolute path to the output healpix mock

    shape_dir: string
        Directory storing files with halo-shape information

    redshift_list : list
        List of length num_snaps storing the value of the redshifts
        in the target halo lightcone cutout

    commit_hash : string
        Commit hash of the version of the cosmodc2 repo used when
        calling this function.

        After updating the code repo to the desired version,
        the commit_hash can be determined by navigating to the root
        directory and typing ``git log --pretty=format:'%h' -n 1``

    synthetic_params: dict contains values for
        skip_synthetics: boolean
        Flag to control if ultra-faint synthetics are added to mock
        synthetic_halo_minimum_mass: float
        Minimum value of log_10 of synthetic halo mass
        num_synthetic_gal_ratio: float
        Ratio to control number of synthetic galaxies generated
        randomize_redshift_synthetic: boolean
        Flag to control if noise is added to redshifts in UM snapshot

    SED_pars: dict containing values for SED choices read in from configuration file
        values supplied by calling script

    shear_params: dict containing values for shear choices
        add_dummy_shears: boolean

    mstar_min: stellar mass cut for synthetic galaxies (not used in image simulations)

    mass_match_noise: noise added to log of source halo masses to randomize the match
        to target halos

    versionMajor: int
        major version number

    versionMinor: int
        minor version number

    versionMinorMinor: int
        minor.minor version number

    Returns
    -------
    None

    """

    from .constants import SED_params_singlemet as SED_params

    output_mock = {}
    gen = zip(umachine_mstar_ssfr_mock_fname_list, redshift_list, snapshots)
    start_time = time()
    process = psutil.Process(os.getpid())

    assert len(cosmological_params) > 0, "No cosmology parameters supplied"
    H0 = cosmological_params["H0"]
    OmegaM = cosmological_params["OmegaM"]
    OmegaB = cosmological_params["OmegaB"]
    w0 = cosmological_params["w0"]
    wa = cosmological_params["wa"]
    print(
        "Cosmology Parameters:\n",
        "H0: {:.2g}, OmegaM: {:.3g}, OmegaB: {:.3g}\n".format(H0, OmegaM, OmegaB),
        "w0: {:.1g} wa: {:.1g}".format(w0, wa),
    )

    cosmology = FlatLambdaCDM(H0=H0, Om0=OmegaM, Ob0=OmegaB)

    #  determine number of healpix cutout to use as offset for galaxy ids
    output_mock_basename = os.path.basename(output_color_mock_fname)
    file_ids = [
        int(d) for d in re.findall(r"\d+", os.path.splitext(output_mock_basename)[0])
    ]

    cutout_number_true = file_ids[-1]
    z_range_id = file_ids[-3]  # 3rd-last digits in filename

    cutout_number = cutout_number_true  # used for output
    galaxy_id_offset = int(
        cutout_number_true * cutout_id_offset + z_offsets_not_im[str(Nside)][z_range_id]
    )
    halo_id_cutout_offset = int(cutout_number_true * cutout_id_offset_halo)

    #  determine seed from output filename
    seed = get_random_seed(output_mock_basename)

    #  determine maximum redshift and volume covered by catalog
    redshift_max = [float(k) for k, v in z2ts.items() if int(v) == snapshot_min][0]
    Vtotal = cosmology.comoving_volume(redshift_max).value

    # determine total number of synthetic galaxies for arbitrary healpixel for
    # full z range
    synthetic_number = int(Ntotal_synthetics / nhpx_total)
    #  number for healpixels straddling the edge of the octant will be adjusted later
    # initialize previous redshift for computing synthetic galaxy distributions
    previous_redshift = get_snap_redshift_min(z2ts, snapshots)

    #  initialize book-keeping variables
    fof_halo_mass_max = 0.0
    Ngals_total = 0

    print("\nStarting snapshot processing")
    print("Using initial seed = {}".format(seed))
    print("Using nside = {}".format(Nside))
    print("Maximum redshift for catalog = {}".format(redshift_max))
    print("Minimum redshift for catalog = {}".format(previous_redshift))
    print(
        "Writing catalog version number {}.{}.{}".format(
            versionMajor, versionMinor, versionMinorMinor
        )
    )
    print("\nUsing halo-id offset = {}".format(halo_id_offset))
    print(
        "Using galaxy-id offset = {} for cutout number {}".format(
            galaxy_id_offset, cutout_number_true
        )
    )

    if synthetic_params and not synthetic_params["skip_synthetics"]:
        synthetic_halo_minimum_mass = synthetic_params["synthetic_halo_minimum_mass"]
        synthetic_number = synthetic_params["synthetic_number"]
        randomize_redshift_synthetic = synthetic_params["randomize_redshift_synthetic"]
        print("Synthetic-halo minimum mass =  {}".format(synthetic_halo_minimum_mass))
        print("Number of synthetic ultra-faint galaxies = {}".format(synthetic_number))
        print("Randomize synthetic redshifts = {}".format(randomize_redshift_synthetic))
    else:
        print("Not adding synthetic galaxies")

    # initialize SED parameters
    if SED_pars:
        # save any supplied parameters from function call
        for k, v in SED_pars.items():
            SED_params[k] = v

    T0 = cosmology.age(SED_params["z0"]).value
    SED_params["LGT0"] = np.log10(T0)
    print(
        "\nUsing SED parameters:\n...{}".format(
            ",\n...".join([": ".join([k, str(v)]) for k, v in SED_params.items()])
        )
    )

    dsps_data_DRN = SED_params["dsps_data_dirname"]
    dsps_data_fn = SED_params["dsps_data_filename"]
    # get ssp_wave, ssp_flux, lg_age_gyr
    ssp_data_singlemet = load_ssp_templates_singlemet(
        os.path.join(dsps_data_DRN, dsps_data_fn)
    )
    # enforce that single-metallicity SSPs were in fact loaded
    assert "ssp_lgmet" not in ssp_data_singlemet._fields
    ssp_wave = ssp_data_singlemet.ssp_wave
    ssp_flux = ssp_data_singlemet.ssp_flux
    filter_data = assemble_filter_data(dsps_data_DRN, SED_params["filters"])
    filter_waves, filter_trans, filter_keys = get_filter_wave_trans(filter_data)
    print(
        "\nUsing filters and bands: {} ({} bands)".format(
            ", ".join(filter_keys), len(filter_keys)
        )
    )

    # generate precomputed ssp tables
    min_snap = 0 if len(healpix_data[snapshots[0]]["a"]) > 0 else 1
    zmin = 1.0 / np.max(healpix_data[snapshots[min_snap]]["a"][()]) - 1.0
    zmax = 1.0 / np.min(healpix_data[snapshots[-1]]["a"][()]) - 1.0
    dz = float(SED_params["dz"])
    z_min = zmin - dz if (zmin - dz) > 0 else zmin / 2  # ensure z_min is > 0
    z_max = zmax + dz
    n_z_table = int(np.ceil((z_max - z_min) / dz))
    ssp_z_table = np.linspace(z_min, z_max, n_z_table)
    msg = "\nComputing ssp tables for {} z values: {:.2f} < z < {:.2f} (dz={:.2f})\n"
    print(msg.format(n_z_table, z_min, z_max, dz))
    ssp_restmag_table = precompute_ssp_restmags_singlemet(
        ssp_wave, ssp_flux, filter_waves, filter_trans
    )
    ssp_obsmag_table = precompute_ssp_obsmags_on_z_table_singlemet(
        ssp_wave,
        ssp_flux,
        filter_waves,
        filter_trans,
        ssp_z_table,
        OmegaM,
        cosmological_params["w0"],
        cosmological_params["wa"],
        H0 / 100.0,
    )

    # save in SED_params for passing to other modules
    SED_params["ssp_z_table"] = ssp_z_table
    SED_params["ssp_lg_age_gyr"] = ssp_data_singlemet.ssp_lg_age_gyr
    SED_params["ssp_restmag_table"] = ssp_restmag_table
    SED_params["ssp_obsmag_table"] = ssp_obsmag_table
    SED_params["filter_keys"] = filter_keys
    SED_params["filter_waves"] = filter_waves
    SED_params["filter_trans"] = filter_trans

    for k in SED_params["xkeys"]:
        dims = (
            SED_params[k].shape
            if not isinstance(SED_params[k], list)
            else len(SED_params[k])
        )
        print("...Saving {} to SED_params with shape: {}".format(k, dims))

    default_list = [k for k in SED_params.values() if type(k) is str and "default" in k]
    for k in default_list:  # placeholder for better code
        SED_params["lgfburst_pop_u_params"] = DEFAULT_LGFBURST_U_PARAMS
        SED_params["burstshapepop_u_params"] = DEFAULT_BURSTSHAPE_U_PARAMS
        SED_params["lgav_dust_u_params"] = DEFAULT_LGAV_U_PARAMS
        SED_params["dust_delta_u_params"] = DEFAULT_DUST_DELTA_U_PARAMS
        SED_params["fracuno_pop_u_params"] = DEFAULT_BORIS_U_PARAMS
        SED_params["lgmet_params"] = list(DEFAULT_MET_PDICT.values())

    roman_rubin_list = [
        k.split("roman_rubin_2023/")[1]
        for k in SED_params.values()
        if type(k) is str and "roman_rubin" in k
    ]

    for k in roman_rubin_list:
        if "lgfburst_u_params" in k:
            SED_params["lgfburst_pop_u_params"] = get_sed_model_params(
                SED_params["param_data_dirname"], SED_params["lgfburst_fname"]
            )
        if "burstshape_u_params" in k:
            SED_params["burstshapepop_u_params"] = get_sed_model_params(
                SED_params["param_data_dirname"], SED_params["burstshape_fname"]
            )
        if "lgav_dust_u_params" in k:
            SED_params["lgav_dust_u_params"] = get_sed_model_params(
                SED_params["param_data_dirname"], SED_params["lgav_dust_fname"]
            )
        if "delta_dust_u_params" in k:
            SED_params["delta_dust_u_params"] = get_sed_model_params(
                SED_params["param_data_dirname"], SED_params["delta_dust_fname"]
            )
        if "funo_dust_u_params" in k:
            SED_params["fracuno_pop_u_params"] = get_sed_model_params(
                SED_params["param_data_dirname"], SED_params["fracuno_pop_fname"]
            )
        if "lgmet_params" in k:
            SED_params["lgmet_params"] = get_sed_model_params(
                SED_params["param_data_dirname"], SED_params["lgmet_fname"]
            )

    print()
    model_keys = [k for k in SED_params.keys() if "_model" in k]
    for key in model_keys:
        # parse column name to extract filter, frame and band
        _res = SED_params[key].split("_")
        if len(_res) >= 2:
            filt_req = SED_params[key].split("_")[0]
            frame_req = SED_params[key].split("_")[1]
            band_req = SED_params[key].split("_")[-1].lower()
            model_req = [
                k for k in SED_params["filter_keys"] if filt_req in k and band_req in k
            ]
            if len(model_req) == 1 and frame_req in SED_params["frames"]:
                print("...Using {} for galaxy-{}".format(SED_params[key], key))
            else:
                print("...{} not available for galaxy-{}".format(SED_params[key], key))
                SED_params[key] = None  # filter not available; overwrite key
        else:
            if "skip" not in SED_params[key]:
                print(
                    "...incorrect option {} for galaxy-{}".format(SED_params[key], key)
                )
            print("...Skipping galaxy-{}".format(key))
            SED_params[key] = None  # filter not available; overwrite key

    t_table = np.linspace(SED_params["t_table_0"], T0, SED_params["N_t_table"])
    SED_params["t_table"] = t_table

    for a, b, c in gen:
        umachine_mock_fname = a
        redshift = b
        snapshot = c
        halo_unique_id = int(halo_id_cutout_offset + int(snapshot))
        print(
            "\n...Using halo_unique id = {} for snapshot {}".format(
                halo_unique_id, snapshot
            )
        )

        new_time_stamp = time()

        #  seed should be changed for each new shell
        seed = seed + 2

        #  check for halos in healpixel
        if len(healpix_data[snapshot]["id"]) == 0:
            output_mock[snapshot] = {}
            print("\n...skipping empty snapshot {}".format(snapshot))
            continue

        #  Get galaxy properties from UM catalogs and target halo properties
        print("\n...loading z = {0:.2f} galaxy catalog into memory".format(redshift))
        mock = Table.read(umachine_mock_fname, path="data")
        print(".....Number of available UM galaxies: {}".format(len(mock)))
        # print('.....UM catalog colnames: {}'.format(', '.join(mock.colnames)))

        # Assemble table of target halos
        target_halos = get_astropy_table(
            healpix_data[snapshot], halo_unique_id=halo_unique_id
        )
        fof_halo_mass_max = max(
            np.max(target_halos[fof_halo_mass].quantity.value), fof_halo_mass_max
        )

        ################################################################################
        # generate halo shapes
        ################################################################################
        print("\n...Generating halo shapes")
        b_to_a, c_to_a, e, p = monte_carlo_halo_shapes(
            np.log10(target_halos[fof_halo_mass])
        )
        target_halos["halo_ellipticity"] = e
        target_halos["halo_prolaticity"] = p
        spherical_halo_radius = halo_mass_to_halo_radius(
            target_halos[fof_halo_mass], WMAP7, redshift, "vir"
        )
        target_halos["axis_A_length"] = (
            1.5 * spherical_halo_radius
        )  # crude fix for B and C shrinking
        target_halos["axis_B_length"] = b_to_a * target_halos["axis_A_length"]
        target_halos["axis_C_length"] = c_to_a * target_halos["axis_A_length"]

        nvectors = len(target_halos)
        rng = np.random.RandomState(seed)
        random_vectors = rng.uniform(-1, 1, nvectors * 3).reshape((nvectors, 3))
        axis_A = normalized_vectors(random_vectors) * target_halos[
            "axis_A_length"
        ].reshape((-1, 1))
        target_halos["axis_A_x"] = axis_A[:, 0]
        target_halos["axis_A_y"] = axis_A[:, 1]
        target_halos["axis_A_z"] = axis_A[:, 2]
        # now add halo shape information for those halos with matches in shape files
        print("\n...Matching halo shapes for selected halos")
        shapes = get_halo_shapes(
            snapshot,
            target_halos["fof_halo_id"],
            target_halos["lightcone_replication"],
            shape_dir,
        )
        if shapes:
            target_halos = get_matched_shapes(
                shapes, target_halos, rep_key="lightcone_replication"
            )

        ################################################################################
        #  Galsampler - For every target halo,
        #               find a source halo with closely matching mass
        ################################################################################
        print("\n...Finding halo--halo correspondence with GalSampler")

        #  Set up target and source halo arrays for indices and masses
        target_halo_ids = target_halos["halo_id"]
        log_target_mass = np.log10(target_halos[fof_halo_mass])
        # Ensure that mock uses uber hostid
        mock.rename_column("hostid", "uber_hostid")
        upid, uber_hostid, _ = compute_uber_hostid(mock["upid"], mock["halo_id"])
        if not np.array_equal(mock["upid"], upid):
            print(".....overwriting upid with corrected upid")
            mock["upid"] = upid
        if not np.array_equal(mock["uber_hostid"], uber_hostid):
            print(".....overwriting hostid with corrected uber hostid")
            mock["uber_hostid"] = uber_hostid
        cenmask = mock["upid"] == -1
        # match to host halos not subhalos
        source_halo_ids = mock["uber_hostid"][cenmask]
        assert len(np.unique(source_halo_ids)) == len(
            source_halo_ids
        ), "Source halo ids not unique"

        source_galaxies_host_halo_id = mock["uber_hostid"]
        log_src_mass = np.log10(mock["mp"][cenmask])
        #  Add noise to randomize the selections around the closest source halo match
        noisy_log_src_mass = np.random.normal(loc=log_src_mass, scale=mass_match_noise)

        #  Use GalSampler to calculate the indices of the galaxies that will be selected
        print("...GalSampling z={0:.2f} galaxies to OuterRim halos".format(redshift))
        gs_results = galsample(
            source_galaxies_host_halo_id,
            source_halo_ids,
            target_halo_ids,
            [noisy_log_src_mass],
            [log_target_mass],
        )

        ########################################################################
        # Add synthetic galaxies if requested
        ########################################################################
        if not synthetic_params["skip_synthetics"]:
            mock, synthetic_dict = add_low_mass_synthetic_galaxies(
                mock,
                seed,
                synthetic_halo_minimum_mass,
                redshift,
                synthetic_number,
                randomize_redshift_synthetic,
                previous_redshift,
                Vtotal,
                cosmology,
                mstar_min,
            )
        else:
            synthetic_dict = {}

        ########################################################################
        #  Assemble the output mock by snapshot
        ########################################################################

        print("\n...Building output snapshot mock for snapshot {}".format(snapshot))
        output_mock[snapshot] = build_output_snapshot_mock(
            ssp_data_singlemet,
            float(redshift),
            mock,
            target_halos,
            gs_results,
            galaxy_id_offset,
            synthetic_dict,
            Nside_sky,
            cutout_number_true,
            float(previous_redshift),
            cosmology,
            w0,
            wa,
            volume_minx=volume_minx,
            SED_params=SED_params,
            volume_miny=volume_miny,
            volume_maxz=volume_maxz,
            seed=seed,
            shear_params=shear_params,
            halo_unique_id=halo_unique_id,
            redshift_method="galaxy",
        )
        galaxy_id_offset = galaxy_id_offset + len(
            output_mock[snapshot]["galaxy_id"]
        )  # increment offset

        # check that offset is within index bounds
        galaxy_id_bound = (
            cutout_number * cutout_id_offset
            + z_offsets_not_im[str(Nside)][z_range_id + 1]
        )
        if galaxy_id_offset > galaxy_id_bound:
            print(
                "...Warning: galaxy_id bound of {} exceeded for snapshot {}".format(
                    galaxy_id_bound, snapshot
                )
            )

        Ngals_total += len(output_mock[snapshot]["galaxy_id"])
        print(
            "...Saved {} galaxies to dict".format(
                len(output_mock[snapshot]["galaxy_id"])
            )
        )
        previous_redshift = redshift  # update for next snap

        # do garbage collection
        gc.collect()

        time_stamp = time()
        msg = "\nLightcone-shell runtime = {0:.2f} minutes"
        print(msg.format((time_stamp - new_time_stamp) / 60.0))

        mem = "Memory usage =  {0:.2f} GB"
        print(mem.format(process.memory_info().rss / 1.0e9))

        mem = "Thread count =  {}"
        print(mem.format(threading.active_count()))
        for env in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            print("Checking {} = {}".format(env, os.environ.get(env)))

    ########################################################################
    #  Write the output mock to disk
    ########################################################################
    if len(output_mock) > 0:
        check_time = time()
        write_output_mock_to_disk(
            output_color_mock_fname,
            output_mock,
            commit_hash,
            seed,
            synthetic_params,
            cutout_number_true,
            Nside,
            cosmology,
            versionMajor=versionMajor,
            versionMinor=versionMinor,
            versionMinorMinor=versionMinorMinor,
        )
        print(
            "...time to write mock to disk = {:.2f} minutes".format(
                (time() - check_time) / 60.0
            )
        )

    print(
        "Maximum halo mass for {} ={}\n".format(output_mock_basename, fof_halo_mass_max)
    )
    print("Number of galaxies for {} ={}\n".format(output_mock_basename, Ngals_total))

    time_stamp = time()
    msg = "\nEnd-to-end runtime = {0:.2f} minutes\n"
    print(msg.format((time_stamp - start_time) / 60.0))


# reduce max seed by 200 to allow for 60 light-cone shells
def get_random_seed(filename, seed_max=4294967095):
    import hashlib

    s = hashlib.md5(filename.encode("utf-8")).hexdigest()
    seed = int(s, 16)

    #  enforce seed is below seed_max and odd
    seed = seed % seed_max
    if seed % 2 == 0:
        seed = seed + 1
    return seed


def get_volume_factor(redshift, previous_redshift, Vtotal, cosmology):
    Vshell = (
        cosmology.comoving_volume(float(redshift)).value
        - cosmology.comoving_volume(float(previous_redshift)).value
    )
    return Vshell / Vtotal


def get_sed_model_params(param_dir, param_file):
    param_filename = os.path.join(param_dir, param_file)
    with open(param_filename) as fh:
        params = [x.split() for x in fh.readlines()]
        print("\nReading model parameters from file {}".format(param_filename))
    keys = tuple([p[0] for p in params])
    values = tuple([float(p[1]) for p in params])
    param_dict = dict(zip(keys, values))
    for k, v in param_dict.items():
        print("....{} = {:.4f}".format(k, v))

    return param_dict


def get_astropy_table(table_data, halo_unique_id=0, check=False, cosmology=None):
    """ """
    t = Table()
    for k in table_data.keys():
        t[k] = table_data[k]

    t.rename_column("id", "fof_halo_id")
    t.rename_column("rot", "lightcone_rotation")
    t.rename_column("rep", "lightcone_replication")
    t["halo_redshift"] = 1 / t["a"] - 1.0

    t["halo_id"] = (
        np.arange(len(table_data["id"])) * halo_id_offset + halo_unique_id
    ).astype(int)

    #  rename column mass if found
    if mass in t.colnames:
        t.rename_column(mass, fof_halo_mass)
    elif fof_mass in t.colnames:
        t.rename_column(fof_mass, fof_halo_mass)
    else:
        print("  Warning; halo mass or fof_mass not found")

    #  check sod information and clean bad values
    if sod_mass in t.colnames:
        mask_valid = t[sod_mass] > 0
        mask = mask_valid & (t[sod_mass] < m_particle_1000)
        # overwrite
        for cn in ["sod_cdelta", "sod_cdelta_error", sod_mass, "sod_radius"]:
            t[cn][mask] = -1

        if np.count_nonzero(mask) > 0:
            print(
                ".....Overwrote {}/{} SOD quantities failing {:.2g} mass cut".format(
                    np.count_nonzero(mask),
                    np.count_nonzero(mask_valid),
                    m_particle_1000,
                )
            )

    if check and cosmology is not None:
        #  compute comoving distance from z and from position
        r = np.sqrt(t["x"] * t["x"] + t["y"] * t["y"] + t["z"] * t["z"])
        comoving_distance = (
            cosmology.comoving_distance(t["halo_redshift"]) * cosmology.H0.value / 100.0
        )
        print("r == comoving_distance(z) is {}", np.isclose(r, comoving_distance))

    print(
        "...Number of target halos to populate with galaxies: {}".format(
            len(t["halo_id"])
        )
    )

    return t


def add_low_mass_synthetic_galaxies(
    mock,
    seed,
    synthetic_halo_minimum_mass,
    redshift,
    synthetic_number,
    randomize_redshift_synthetic,
    previous_redshift,
    Vtotal,
    cosmology,
    mstar_min,
    use_substeps_synthetic=False,
    nzdivs=6,
):
    #  Correct stellar mass for low-mass subhalos and create synthetic mpeak
    print(".....correcting low mass mpeak and assigning synthetic mpeak values")

    #  First generate the appropriate number of synthetic galaxies for the snapshot
    mpeak_synthetic_snapshot = 10 ** synthetic_logmpeak(
        mock["mpeak"], seed=seed, desired_logm_completeness=synthetic_halo_minimum_mass
    )
    print(".....assembling {} synthetic galaxies".format(len(mpeak_synthetic_snapshot)))

    # Add call to map_mstar_onto_lowmass_extension function after
    # pre-determining low-mass slope
    print(".....re-assigning low-mass mstar values")
    new_mstar_real, mstar_synthetic_snapshot = map_mstar_onto_lowmass_extension(
        mock["mpeak"], mock["obs_sm"], mpeak_synthetic_snapshot
    )
    diff = np.equal(new_mstar_real, mock["obs_sm"])
    print(
        ".......changed {}/{} M* values; max/min new log(M*) {:.2f}/{:.2f}".format(
            np.count_nonzero(~diff),
            len(diff),
            np.max(np.log10(new_mstar_real[~diff])),
            np.min(np.log10(new_mstar_real[~diff])),
        )
    )
    mock["obs_sm"] = new_mstar_real
    mstar_mask = np.isclose(mstar_synthetic_snapshot, 0.0)
    if np.sum(mstar_mask) > 0:
        print(
            ".....Warning: Number of synthetics with zero mstar = {}".format(
                np.sum(mstar_mask)
            )
        )

    #  Assign diffstar parameters to synthetic low-mass galaxies
    print("TBD: get diffstar parameters for synthetic galaxies")

    #  Now downsample the synthetic galaxies to adjust for volume of lightcone shell
    #  desired number = synthetic_number*comoving_vol(snapshot)/comoving_vol(healpixel)
    volume_factor = get_volume_factor(redshift, previous_redshift, Vtotal, cosmology)
    num_selected_synthetic = int(synthetic_number * volume_factor)
    num_synthetic_gals_in_snapshot = len(mpeak_synthetic_snapshot)
    synthetic_indices = np.arange(0, num_synthetic_gals_in_snapshot).astype(int)
    with NumpyRNGContext(seed):
        selected_synthetic_indices = np.random.choice(
            synthetic_indices, size=num_selected_synthetic, replace=False
        )
    msg = ".....down-sampling synthetic galaxies with volume factor {} to yield {}"
    print(
        "{} selected synthetics".format(
            msg.format(volume_factor, num_selected_synthetic)
        )
    )
    mstar_synthetic = mstar_synthetic_snapshot[selected_synthetic_indices]
    #  Apply additional M* cut to reduce number of synthetics for 5000 sq. deg. catalog
    if mstar_min > 0:
        mstar_mask = mstar_synthetic > mstar_min
        msg = ".....removing synthetics with M* < {:.1e} to yield {} total synthetics"
        print(msg.format(mstar_min, np.count_nonzero(mstar_mask)))

    mstar_synthetic = mstar_synthetic[mstar_mask]
    mpeak_synthetic = mpeak_synthetic_snapshot[selected_synthetic_indices][mstar_mask]
    synthetic_dict = dict(
        mpeak=mpeak_synthetic,
        obs_sm=mstar_synthetic,
    )

    return mock, synthetic_dict


def build_output_snapshot_mock(
    ssp_data_singlemet,
    snapshot_redshift,
    umachine,
    target_halos,
    gs_results,
    galaxy_id_offset,
    synthetic_dict,
    Nside,
    cutout_number_true,
    previous_redshift,
    cosmology,
    w0,
    wa,
    volume_minx=0.0,
    volume_miny=0.0,
    volume_maxz=0.0,
    SED_params={},
    seed=41,
    shear_params={},
    mah_keys="mah_keys",
    ms_keys="ms_keys",
    q_keys="q_keys",
    mah_pars="mah_params",
    ms_pars="ms_params",
    q_pars="q_params",
    halo_unique_id=0,
    redshift_method="galaxy",
    source_galaxy_tag="um_source_galaxy_",
    bulge_frac="bulge_frac",
):
    """
    Collect the GalSampled snapshot mock into an astropy table

    Parameters
    ----------
    snapshot_redshift : float
        Float of the snapshot redshift

    umachine : astropy.table.Table
        Astropy Table of shape (num_source_gals, )
        storing the UniverseMachine snapshot mock

    target_halos : astropy.table.Table
        Astropy Table of shape (num_target_halos, )
        storing the target halo catalog

    gs_results: named ntuple
        Named ntuple returned by galsample containing 3 arrays
        of shape (num_target_gals, )
        storing integers valued between [0, num_source_gals)

    commit_hash : string
        Commit hash of the version of the code repo used when
        calling this function.

        After updating the code repo to the desired version,
        the commit_hash can be determined by navigating to the root
        directory and typing ``git log --pretty=format:'%h' -n 1``

    Returns
    -------
    dc2 : astropy.table.Table
        Astropy Table of shape (num_target_gals, )
        storing the GalSampled galaxy catalog
    """
    dc2 = Table()

    # unpack galsampler results
    galaxy_indices = gs_results.target_gals_selection_indx
    target_gals_target_halo_ids = gs_results.target_gals_target_halo_ids
    target_gals_source_halo_ids = gs_results.target_gals_source_halo_ids

    dc2["source_halo_uber_hostid"] = target_gals_source_halo_ids
    dc2["target_halo_id"] = target_gals_target_halo_ids

    # save target halo information into mock
    # compute richness
    # tgt_unique, tgt_inv, tgt_counts = np.unique(target_gals_target_halo_ids,
    #                                             return_inverse=True,
    #                                             return_counts=True)
    # dc2['richness'] = tgt_counts[tgt_inv]
    #
    # Method 1: use unique arrays to get values
    # hunique, hidx = np.unique(target_halos['halo_id'], return_index=True)
    # reconstruct mock arrays using target_halos[name][hidx][tgt_inv]

    # Method 2: cross-match to get target halo information
    idxA, idxB = crossmatch(target_gals_target_halo_ids, target_halos["halo_id"])
    msg = "target IDs do not match!"
    assert np.all(dc2["target_halo_id"][idxA] == target_halos["halo_id"][idxB]), msg

    for col in ["lightcone_rotation", "lightcone_replication"]:
        dc2[col] = 0.0
        dc2[col][idxA] = target_halos[col][idxB]

    dc2["target_halo_fof_halo_id"] = 0.0
    dc2["target_halo_redshift"] = 0.0
    dc2["target_halo_x"] = 0.0
    dc2["target_halo_y"] = 0.0
    dc2["target_halo_z"] = 0.0
    dc2["target_halo_vx"] = 0.0
    dc2["target_halo_vy"] = 0.0
    dc2["target_halo_vz"] = 0.0

    dc2["target_halo_fof_halo_id"][idxA] = target_halos["fof_halo_id"][idxB]
    dc2["target_halo_redshift"][idxA] = target_halos["halo_redshift"][idxB]
    dc2["target_halo_x"][idxA] = target_halos["x"][idxB]
    dc2["target_halo_y"][idxA] = target_halos["y"][idxB]
    dc2["target_halo_z"][idxA] = target_halos["z"][idxB]

    dc2["target_halo_vx"][idxA] = target_halos["vx"][idxB]
    dc2["target_halo_vy"][idxA] = target_halos["vy"][idxB]
    dc2["target_halo_vz"][idxA] = target_halos["vz"][idxB]

    dc2["target_halo_mass"] = 0.0
    dc2["target_halo_mass"][idxA] = target_halos["fof_halo_mass"][idxB]

    dc2["target_halo_ellipticity"] = 0.0
    dc2["target_halo_ellipticity"][idxA] = target_halos["halo_ellipticity"][idxB]

    dc2["target_halo_prolaticity"] = 0.0
    dc2["target_halo_prolaticity"][idxA] = target_halos["halo_prolaticity"][idxB]

    dc2["target_halo_axis_A_length"] = 0.0
    dc2["target_halo_axis_B_length"] = 0.0
    dc2["target_halo_axis_C_length"] = 0.0
    dc2["target_halo_axis_A_length"][idxA] = target_halos["axis_A_length"][idxB]
    dc2["target_halo_axis_B_length"][idxA] = target_halos["axis_B_length"][idxB]
    dc2["target_halo_axis_C_length"][idxA] = target_halos["axis_C_length"][idxB]

    dc2["target_halo_axis_A_x"] = 0.0
    dc2["target_halo_axis_A_y"] = 0.0
    dc2["target_halo_axis_A_z"] = 0.0
    dc2["target_halo_axis_A_x"][idxA] = target_halos["axis_A_x"][idxB]
    dc2["target_halo_axis_A_y"][idxA] = target_halos["axis_A_y"][idxB]
    dc2["target_halo_axis_A_z"][idxA] = target_halos["axis_A_z"][idxB]

    # add SOD information from target_halo table
    dc2["sod_halo_cdelta"] = 0.0
    dc2["sod_halo_cdelta_error"] = 0.0
    dc2["sod_halo_mass"] = 0.0
    dc2["sod_halo_radius"] = 0.0
    dc2["sod_halo_cdelta"][idxA] = target_halos["sod_cdelta"][idxB]
    dc2["sod_halo_cdelta_error"][idxA] = target_halos["sod_cdelta_error"][idxB]
    dc2["sod_halo_mass"][idxA] = target_halos["sod_mass"][idxB]
    dc2["sod_halo_radius"][idxA] = target_halos["sod_radius"][idxB]

    #  Here the host_centric_xyz_vxvyvz in umachine should be overwritten
    #  Then we can associate x <--> A, y <--> B, z <--> C and then apply
    #  a random rotation
    #  It will be important to record the true direction of the major axis as a
    #  stored column

    source_galaxy_prop_keys = (
        "mp",
        "vmp",
        "rvir",
        "upid",
        "host_rvir",
        "host_mp",
        "host_rvir",
        "halo_id",
        "has_fit",
        "is_main_branch",
        "obs_sm",
        "obs_sfr",
        "sfr_percentile",
    )
    source_galaxy_pv_keys = (
        "host_dx",
        "host_dy",
        "host_dz",
        "host_dvx",
        "host_dvy",
        "host_dvz",
    )
    # check for no-fit replacement
    if "nofit_replace" in umachine.colnames:
        source_galaxy_prop_keys = source_galaxy_prop_keys + ("nofit_replace",)
    SFH_param_keys = SED_params[mah_keys] + SED_params[ms_keys] + SED_params[q_keys]

    source_galaxy_keys = (
        source_galaxy_pv_keys + source_galaxy_prop_keys + tuple(SFH_param_keys)
    )

    for key in source_galaxy_keys:
        newkey = source_galaxy_tag + key if key in source_galaxy_prop_keys else key
        try:
            dc2[newkey] = umachine[key][galaxy_indices]
        except KeyError:
            msg = (
                "The build_output_snapshot_mock function was passed a umachine mock\n"
                "that does not contain the ``{0}`` key"
            )
            raise KeyError(msg.format(key))

    # remap M* for high-mass halos
    max_umachine_halo_mass = np.max(umachine["mp"])
    ultra_high_mvir_halo_mask = (dc2[source_galaxy_tag + "upid"] == -1) & (
        dc2["target_halo_mass"] > max_umachine_halo_mass
    )
    num_to_remap = np.count_nonzero(ultra_high_mvir_halo_mask)
    if num_to_remap > 0:
        print(
            ".....remapping stellar mass of {0} BCGs in ultra-massive halos".format(
                num_to_remap
            )
        )

        halo_mass_array = dc2["target_halo_mass"][ultra_high_mvir_halo_mask]
        mpeak_array = dc2[source_galaxy_tag + "mp"][ultra_high_mvir_halo_mask]
        mhalo_ratio = halo_mass_array / mpeak_array
        mstar_array = dc2[source_galaxy_tag + "obs_sm"][ultra_high_mvir_halo_mask]
        redshift_array = dc2["target_halo_redshift"][ultra_high_mvir_halo_mask]
        upid_array = dc2[source_galaxy_tag + "upid"][ultra_high_mvir_halo_mask]

        assert np.shape(halo_mass_array) == (
            num_to_remap,
        ), "halo_mass_array has shape = {0}".format(np.shape(halo_mass_array))
        assert np.shape(mstar_array) == (
            num_to_remap,
        ), "mstar_array has shape = {0}".format(np.shape(mstar_array))
        assert np.shape(redshift_array) == (
            num_to_remap,
        ), "redshift_array has shape = {0}".format(np.shape(redshift_array))
        assert np.shape(upid_array) == (
            num_to_remap,
        ), "upid_array has shape = {0}".format(np.shape(upid_array))
        assert np.all(
            mhalo_ratio >= 1
        ), "Bookkeeping error: all values of mhalo_ratio ={0} should be >= 1".format(
            mhalo_ratio
        )

        obs_sm_key = source_galaxy_tag + "obs_sm"
        halo_id_key = source_galaxy_tag + "halo_id"
        dc2[obs_sm_key][ultra_high_mvir_halo_mask] = mstar_array * (mhalo_ratio**0.5)
        idx = np.argmax(dc2[obs_sm_key])
        halo_id_most_massive = dc2[halo_id_key][idx]
        assert (
            dc2[obs_sm_key][idx] < 10**13.5
        ), "halo_id = {0} has stellar mass {1:.3e}".format(
            halo_id_most_massive, dc2[obs_sm_key][idx]
        )

    # generate triaxial satellite distributions based on halo shapes
    satmask = dc2[source_galaxy_tag + "upid"] != -1
    nsats = np.count_nonzero(satmask)
    host_conc = 5.0
    if nsats > 0:
        host_Ax = dc2["target_halo_axis_A_x"][satmask]
        host_Ay = dc2["target_halo_axis_A_y"][satmask]
        host_Az = dc2["target_halo_axis_A_z"][satmask]
        b_to_a = (
            dc2["target_halo_axis_B_length"][satmask]
            / dc2["target_halo_axis_A_length"][satmask]
        )
        c_to_a = (
            dc2["target_halo_axis_C_length"][satmask]
            / dc2["target_halo_axis_A_length"][satmask]
        )
        host_dx, host_dy, host_dz = generate_triaxial_satellite_distribution(
            host_conc, host_Ax, host_Ay, host_Az, b_to_a, c_to_a
        )
        dc2["host_dx"][satmask] = host_dx
        dc2["host_dy"][satmask] = host_dy
        dc2["host_dz"][satmask] = host_dz

    # save positions and velocities
    dc2["x"] = dc2["target_halo_x"] + dc2["host_dx"]
    dc2["vx"] = dc2["target_halo_vx"] + dc2["host_dvx"]

    dc2["y"] = dc2["target_halo_y"] + dc2["host_dy"]
    dc2["vy"] = dc2["target_halo_vy"] + dc2["host_dvy"]

    dc2["z"] = dc2["target_halo_z"] + dc2["host_dz"]
    dc2["vz"] = dc2["target_halo_vz"] + dc2["host_dvz"]

    print(
        ".....number of galaxies before adding synthetic satellites = {}".format(
            len(dc2[source_galaxy_tag + "halo_id"])
        )
    )

    # add synthetic cluster galaxies
    fake_cluster_sats = model_synthetic_cluster_satellites(
        dc2,
        Lbox=0.0,
        host_conc=host_conc,
        SFH_keys=list(SFH_param_keys),
        source_halo_mass_key=source_galaxy_tag + "host_mp",
        source_halo_id_key=source_galaxy_tag + "halo_id",
        upid_key=source_galaxy_tag + "upid",
        tri_axial_positions=True,
        source_galaxy_tag=source_galaxy_tag,
    )  # turn off periodicity
    if len(fake_cluster_sats) > 0:
        print(".....generating and stacking synthetic cluster satellites")
        check_time = time()
        dc2 = vstack((dc2, fake_cluster_sats))
        print(
            ".....time to create {} galaxies in fake_cluster_sats = {:.2f} secs".format(
                len(fake_cluster_sats["target_halo_id"]), time() - check_time
            )
        )
    else:
        print(".....no synthetic cluster satellites required")

    # generate redshifts, ra and dec
    dc2 = get_sky_coords(
        dc2, cosmology, redshift_method, source_galaxy_tag=source_galaxy_tag
    )

    # save number of galaxies in shell
    Ngals = len(dc2["target_halo_id"])

    # generate mags
    dc2 = generate_SEDs(
        ssp_data_singlemet,
        dc2,
        SED_params,
        cosmology,
        w0,
        wa,
        seed,
        snapshot_redshift,
        mah_keys,
        ms_keys,
        q_keys,
        Ngals,
        mah_pars=mah_pars,
        ms_pars=ms_pars,
        q_pars=q_pars,
        source_galaxy_tag=source_galaxy_tag,
    )

    # Add low-mass synthetic galaxies
    if synthetic_dict and len(synthetic_dict["mp"]) > 0:
        check_time = time()
        lowmass_mock = create_synthetic_lowmass_mock_with_centrals(
            umachine,
            dc2,
            synthetic_dict,
            previous_redshift,
            snapshot_redshift,
            cosmology,
            Nside=Nside,
            cutout_id=cutout_number_true,
            H0=cosmology.H0.value,
            volume_minx=volume_minx,
            volume_miny=volume_miny,
            volume_maxz=volume_maxz,
            halo_id_offset=halo_id_offset,
            halo_unique_id=halo_unique_id,
        )
        lowmass_mock = get_sky_coords(lowmass_mock, cosmology, redshift_method="halo")

        if len(lowmass_mock) > 0:
            # astropy vstack pads missing values with zeros in lowmass_mock
            dc2 = vstack((dc2, lowmass_mock))
            msg = ".....time to create {} galaxies in lowmass_mock = {:.2f} secs"
            print(
                msg.format(
                    len(lowmass_mock["target_halo_id"]),
                    time() - check_time,
                )
            )

    # Add shears and magnification
    if shear_params["add_dummy_shears"]:
        print("\n.....adding dummy shears and magnification")
        dc2["shear1"] = np.zeros(Ngals, dtype="f4")
        dc2["shear2"] = np.zeros(Ngals, dtype="f4")
        dc2["magnification"] = np.ones(Ngals, dtype="f4")
        dc2["convergence"] = np.zeros(Ngals, dtype="f4")
    else:
        print("\n.....TBD: add real shears")

    # Add auxiliary quantities for sizes and ellipticities and black-hole masses
    # TBD Need dc2['bulge_to_total_ratio'] for some quantities

    if SED_params["size_model_mag"]:
        size_disk, size_sphere, arcsec_per_kpc = get_galaxy_sizes(
            dc2[SED_params["size_model_mag"]], dc2["redshift"], cosmology
        )
        dc2["spheroidHalfLightRadius"] = size_sphere
        dc2["spheroidHalfLightRadiusArcsec"] = size_sphere * arcsec_per_kpc
        dc2["diskHalfLightRadius"] = size_disk
        dc2["diskHalfLightRadiusArcsec"] = size_disk * arcsec_per_kpc

    if SED_params["ellipticity_model_mag"]:
        pos_angle = np.random.uniform(size=Ngals) * np.pi
        spheroid_ellip_cosmos, disk_ellip_cosmos = monte_carlo_ellipticity_bulge_disk(
            dc2[SED_params["ellipticity_model_mag"]]
        )
        # Returns distortion ellipticity = 1-q^2 / 1+q^2; q=axis ratio
        spheroid_axis_ratio = np.sqrt(
            (1 - spheroid_ellip_cosmos) / (1 + spheroid_ellip_cosmos)
        )
        disk_axis_ratio = np.sqrt((1 - disk_ellip_cosmos) / (1 + disk_ellip_cosmos))
        # Calculate ellipticity from the axis ratios using shear ellipticity e =
        # 1-q / 1+q
        ellip_disk = (1.0 - disk_axis_ratio) / (1.0 + disk_axis_ratio)
        ellip_spheroid = (1.0 - spheroid_axis_ratio) / (1.0 + spheroid_axis_ratio)
        dc2["spheroidAxisRatio"] = np.array(spheroid_axis_ratio, dtype="f4")
        dc2["spheroidEllipticity"] = np.array(ellip_spheroid, dtype="f4")
        dc2["spheroidEllipticity1"] = np.array(
            np.cos(2.0 * pos_angle) * ellip_spheroid, dtype="f4"
        )
        dc2["spheroidEllipticity2"] = np.array(
            np.sin(2.0 * pos_angle) * ellip_spheroid, dtype="f4"
        )
        dc2["diskAxisRatio"] = np.array(disk_axis_ratio, dtype="f4")
        dc2["diskEllipticity"] = np.array(ellip_disk, dtype="f4")
        dc2["diskEllipticity1"] = np.array(
            np.cos(2.0 * pos_angle) * ellip_disk, dtype="f4"
        )
        dc2["diskEllipticity2"] = np.array(
            np.sin(2.0 * pos_angle) * ellip_disk, dtype="f4"
        )
        dc2["positionAngle"] = np.array(pos_angle * 180.0 / np.pi, dtype="f4")
        if bulge_frac in dc2.colnames:
            tot_ellip = (1.0 - dc2[bulge_frac]) * ellip_disk + dc2[
                bulge_frac
            ] * ellip_spheroid
            dc2["totalEllipticity"] = np.array(tot_ellip, dtype="f4")
            dc2["totalAxisRatio"] = np.array(
                (1.0 - tot_ellip) / (1.0 + tot_ellip), dtype="f4"
            )
            dc2["totalEllipticity1"] = np.array(
                np.cos(2.0 * pos_angle) * tot_ellip, dtype="f4"
            )
            dc2["totalEllipticity2"] = np.array(
                np.sin(2.0 * pos_angle) * tot_ellip, dtype="f4"
            )
            # srsc_indx_disk = 1.0*np.ones(lum_disk.size,dtype='f4')
            # srsc_indx_sphere = 4.0*np.ones(lum_disk.size,dtype='f4')
            # srsc_indx_tot = srsc_indx_disk*(1. - dc2['bulge_to_total_ratio'])
            #                 + srsc_indx_sphere*dc2['bulge_to_total_ratio']
            # dc2['diskSersicIndex'] = srsc_indx_disk
            # dc2['spheroidSersicIndex'] = srsc_indx_sphere
            # dc2['totalSersicIndex'] = srsc_indx_tot

    if SED_params["black_hole_model"]:
        percentile_sfr = dc2[source_galaxy_tag + "sfr_percentile"]
        # percentile_sfr = np.random.uniform(size=Ngals)
        dc2["bulge_stellar_mass"] = dc2[bulge_frac] * np.power(10, dc2["logsm_obs"])
        dc2["blackHoleMass"] = monte_carlo_black_hole_mass(dc2["bulge_stellar_mass"])
        eddington_ratio, bh_acc_rate = monte_carlo_bh_acc_rate(
            snapshot_redshift, dc2["blackHoleMass"], percentile_sfr
        )
        dc2["blackHoleAccretionRate"] = bh_acc_rate * 1e9
        dc2["blackHoleEddingtonRatio"] = eddington_ratio

    # Add column for redshifts including peculiar velocities
    _, z_obs, v_pec, _, _, _, _ = pecZ(
        dc2["x"], dc2["y"], dc2["z"], dc2["vx"], dc2["vy"], dc2["vz"], dc2["redshift"]
    )

    dc2["peculiarVelocity"] = np.array(v_pec, dtype="f4")
    dc2.rename_column("redshift", "redshiftHubble")
    dc2["redshift"] = z_obs

    # Galaxy ids
    dc2["galaxy_id"] = np.arange(
        galaxy_id_offset, galaxy_id_offset + len(dc2["target_halo_id"])
    ).astype(int)
    print(
        "\n.....Min and max galaxy_id = {} -> {}".format(
            np.min(dc2["galaxy_id"]), np.max(dc2["galaxy_id"])
        )
    )

    # convert table to dict
    check_time = time()
    output_dc2 = {}
    for k in dc2.keys():
        output_dc2[k] = dc2[k].quantity.value

    print(".....time to new dict = {:.4f} secs".format(time() - check_time))

    return output_dc2


def generate_SEDs(
    ssp_data_singlemet,
    dc2,
    SED_params,
    cosmology,
    w0,
    wa,
    seed,
    snapshot_redshift,
    mah_keys,
    ms_keys,
    q_keys,
    Ngals,
    mah_pars="mah_params",
    ms_pars="ms_params",
    q_pars="q_params",
    source_galaxy_tag="source_galaxy",
):
    check = validate_SED_params(
        SED_params,
        mah_keys=mah_keys,
        ms_keys=ms_keys,
        q_keys=q_keys,
    )
    assert check, "SED_params does not have required contents"

    dc2 = substitute_SFH_fit_failures(
        dc2,
        SED_params,
        source_galaxy_tag,
        seed,
        cosmology,
        snapshot_redshift,
        mah_keys=mah_keys,
        ms_keys=ms_keys,
        q_keys=q_keys,
    )
    """
    assemble params from UM matches and compute SFH
    """
    _res = get_diff_params(
        dc2,
        mah_keys=SED_params[mah_keys],
        ms_keys=SED_params[ms_keys],
        q_keys=SED_params[q_keys],
    )
    mah_params, ms_params, q_params = _res
    t_obs = cosmology.age(dc2["redshift"]).value

    # get SFH table and observed stellar mass
    sfh_table = sfh_galpop(
        SED_params["t_table"],
        mah_params,
        ms_params,
        q_params,
        lgt0=SED_params["LGT0"],
        fb=FB,
    )
    logsm_obs, sfr_obs = get_logsm_sfr_obs(sfh_table, t_obs, SED_params["t_table"])
    dc2["logsm_obs"] = logsm_obs
    dc2["sfr"] = sfr_obs
    log_ssfr = get_log_safe_ssfr(np.power(10, logsm_obs), sfr_obs)
    dc2["log_ssfr"] = log_ssfr

    fb = cosmology.Ob0 / cosmology.Om0
    cosmo_params = CosmoParams(cosmology.Om0, w0, wa, cosmology.H0.value / 100, fb)

    # compute SEDs
    ran_key = jran.PRNGKey(seed)
    diffskypop_params = DiffskyPopParams(
        np.array(list(SED_params["lgfburst_pop_u_params"].values())),
        np.array(list(SED_params["burstshapepop_u_params"].values())),
        np.array(list(SED_params["lgav_dust_u_params"].values())),
        np.array(list(SED_params["delta_dust_u_params"].values())),
        np.array(list(SED_params["fracuno_pop_u_params"].values())),
        np.array(list(SED_params["lgmet_params"].values())),
    )

    _res = get_diffsky_sed_info_singlemet(
        ran_key,
        dc2["redshift"],
        mah_params,
        ms_params,
        q_params,
        SED_params["ssp_z_table"],
        SED_params["ssp_restmag_table"],
        SED_params["ssp_obsmag_table"],
        ssp_data_singlemet,
        SED_params["t_table"],
        SED_params["filter_waves"],
        SED_params["filter_trans"],
        SED_params["filter_waves"],
        SED_params["filter_trans"],
        diffskypop_params,
        cosmo_params,
    )

    # save quantities to DC2
    dc2 = save_sed_info(dc2, _res, SED_params)

    return dc2


def validate_SED_params(
    SED_params,
    required=[
        "use_diffmah_pop",
        "LGT0",
        "t_table",
        "ssp_z_table",
        "ssp_restmag_table",
        "ssp_obsmag_table",
        "ssp_lg_age_gyr",
        "t_table",
        "filter_waves",
        "filter_trans",
        "lgfburst_pop_u_params",
        "burstshapepop_u_params",
        "lgav_dust_u_params",
        "delta_dust_u_params",
        "fracuno_pop_u_params",
        "lgmet_params",
    ],
    mah_keys="mah_keys",
    ms_keys="ms_keys",
    q_keys="q_keys",
):
    check = True
    for k in required + [mah_keys] + [ms_keys] + [q_keys]:
        if k not in SED_params.keys():
            print(".....Validate SED_params: {} not found".format(k))
            check = False

    return check


def substitute_SFH_fit_failures(
    dc2,
    SED_params,
    source_galaxy_tag,
    seed,
    cosmology,
    snapshot_redshift,
    mah_keys="mah_keys",
    ms_keys="ms_keys",
    q_keys="q_keys",
):
    # check for fit failures
    has_fit = dc2[source_galaxy_tag + "has_fit"] == 1
    # check for replacement
    nfail = np.count_nonzero(~has_fit)
    nmissed = -1
    use_diffmah_pop = SED_params["use_diffmah_pop"]
    if source_galaxy_tag + "nofit_replace" in dc2.colnames:
        nofit_replace = dc2[source_galaxy_tag + "nofit_replace"][~has_fit] == 1
        n_replace = np.count_nonzero(nofit_replace)
        if n_replace > 0:
            msg = ".....Replaced {} diffmah/diffstar fit failures with {}"
            print("{} resampled UM fit successes".format(msg.format(nfail, n_replace)))
        else:
            msg = ".....No replacements required; {} fit failures, {} replacements"
            print(msg.format(nfail, n_replace))
        nmissed = nfail - n_replace
    if nmissed > 0 or (nmissed < 0 and nfail > 0) or (use_diffmah_pop and nfail > 0):
        msg = ".....Replacing parameters for {} fit failures with diffmah{} pop"
        if nmissed > 0 and not use_diffmah_pop:
            failed_mask = ~nofit_replace
            print(".......{}".format(msg.format(nmissed, "/diffstar")))
        else:
            failed_mask = ~has_fit
            txt = "" if use_diffmah_pop else "/diffstar"
            print(".......{}".format(msg.format(nfail, txt)))

        logmh = np.array(np.log10(dc2["target_halo_mass"][failed_mask]))
        logmh = logmh.astype(np.float32)
        ran_key = jran.PRNGKey(seed)
        t_obs = cosmology.age(snapshot_redshift).value
        mc_galpop = mc_diffstarpop(ran_key, t_obs, logmh=logmh)
        mc_mah_params, mc_msk_is_quenched, mc_ms_params, mc_q_params = mc_galpop
        # copy requested mc_params to dc2 table
        key_labels = [mah_keys] if use_diffmah_pop else [mah_keys, ms_keys, q_keys]
        mc_parlist = (
            [mc_mah_params]
            if use_diffmah_pop
            else [mc_mah_params, mc_ms_params, mc_q_params]
        )
        for key_label, mc_params in zip(key_labels, mc_parlist):
            for i, key in enumerate(
                SED_params[key_label]
            ):  # potential bug here if some other subset of fit params selected
                dc2[key][failed_mask] = mc_params[:, i]
                print(".......saving pop model parameters {}".format(key))

    return dc2


def save_sed_info(dc2, _res, SED_params):
    gal_weights, gal_frac_trans_obs, gal_frac_trans_rest = _res[:3]
    gal_att_curve_params = _res[3]
    gal_frac_unobs, gal_fburst, gal_burstshape_params = _res[4:7]
    gal_frac_bulge_t_obs, gal_fbulge_params, gal_fknot = _res[7:10]
    gal_obsmags_nodust, gal_restmags_nodust = _res[10:12]
    gal_obsmags_dust, gal_restmags_dust = _res[12:]

    # add values to catalog
    dc2["dust_eb"] = gal_att_curve_params[:, 0]
    dc2["dust_delta"] = gal_att_curve_params[:, 1]
    dc2["dust_av"] = gal_att_curve_params[:, 2]
    dc2["fburst"] = gal_fburst
    dc2["burstshape_lgyr_peak"] = gal_burstshape_params[:, 0]
    dc2["burstshape_lgyr_max"] = gal_burstshape_params[:, 1]
    dc2["bulge_frac"] = gal_frac_bulge_t_obs
    dc2["fbulge_tcrit"] = gal_fbulge_params[:, 0]
    dc2["fbulge_early"] = gal_fbulge_params[:, 1]
    dc2["fbulge_late"] = gal_fbulge_params[:, 2]
    dc2["fknot"] = gal_fknot

    for dustlabel, results in zip(
        ["", "_nodust"],
        [
            [gal_restmags_dust, gal_obsmags_dust],
            [gal_restmags_nodust, gal_obsmags_nodust],
        ],
    ):
        for fr, vals in zip(["rest", "obs"], results):
            for k in SED_params["filter_keys"]:
                filt = k.split("_")[0]
                band = k.split("_")[1]
                band = band.upper() if fr == "rest" else band
                colname = "{}_{}_{}{}".format(filt, fr, band, dustlabel)
                column = SED_params["filter_keys"].index(k)
                dc2[colname] = vals[:, column]

    return dc2


def get_galaxy_sizes(SDSS_R, redshift, cosmology):
    if len(redshift) > 0:
        arcsec_per_kpc = cosmology.arcsec_per_kpc_proper(redshift).value
    else:
        arcsec_per_kpc = np.zeros(0, dtype=np.float)

    size_disk = mc_size_vs_luminosity_late_type(SDSS_R, redshift)
    size_sphere = mc_size_vs_luminosity_early_type(SDSS_R, redshift)
    return size_disk, size_sphere, arcsec_per_kpc


def get_sky_coords(
    dc2, cosmology, redshift_method="halo", Nzgrid=50, source_galaxy_tag="source_galaxy"
):
    #  compute galaxy redshift, ra and dec
    if redshift_method is not None:
        print(
            "\n.....Generating lightcone redshifts using {} method".format(
                redshift_method
            )
        )
        r = np.sqrt(dc2["x"] * dc2["x"] + dc2["y"] * dc2["y"] + dc2["z"] * dc2["z"])
        mask = r > 5000.0
        if np.sum(mask) > 0:
            print("WARNING: Found {} co-moving distances > 5000".format(np.sum(mask)))

        dc2["redshift"] = dc2["target_halo_redshift"]  # copy halo redshifts to galaxies
        H0 = cosmology.H0.value
        if redshift_method == "galaxy":
            #  generate distance estimates for values between min and max redshifts
            zmin = np.min(dc2["redshift"])
            zmax = np.max(dc2["redshift"])
            zgrid = np.logspace(np.log10(zmin), np.log10(zmax), Nzgrid)
            CDgrid = cosmology.comoving_distance(zgrid) * H0 / 100.0
            #  use interpolation to get redshifts for satellites only
            sat_mask = dc2[source_galaxy_tag + "upid"] != -1
            dc2["redshift"][sat_mask] = np.interp(r[sat_mask], CDgrid, zgrid)

        dc2["dec"] = 90.0 - np.arccos(dc2["z"] / r) * 180.0 / np.pi  # co-latitude
        dc2["ra"] = np.arctan2(dc2["y"], dc2["x"]) * 180.0 / np.pi
        dc2["ra"][(dc2["ra"] < 0)] += 360.0  # force value 0->360

        print(
            ".......min/max z for shell: {:.3f}/{:.3f}".format(
                np.min(dc2["redshift"]), np.max(dc2["redshift"])
            )
        )
    return dc2


def get_skyarea(output_mock, Nside):
    """ """
    import healpy as hp

    #  compute sky area from ra and dec ranges of galaxies
    nominal_skyarea = np.rad2deg(np.rad2deg(4.0 * np.pi / hp.nside2npix(Nside)))
    if Nside > 8:
        skyarea = nominal_skyarea
    else:
        pixels = set()
        for k in output_mock.keys():
            if output_mock[k].has_key("ra") and output_mock[k].has_key("dec"):
                for ra, dec in zip(output_mock[k]["ra"], output_mock[k]["dec"]):
                    pixels.add(hp.ang2pix(Nside, ra, dec, lonlat=True))
        frac = len(pixels) / float(hp.nside2npix(Nside))
        skyarea = frac * np.rad2deg(np.rad2deg(4.0 * np.pi))
        # agreement to about 1 sq. deg.
        if np.isclose(skyarea, nominal_skyarea, rtol=0.02):
            print(" Replacing calculated sky area {} with nominal_area".format(skyarea))
            skyarea = nominal_skyarea
        if np.isclose(
            skyarea, nominal_skyarea / 2.0, rtol=0.01
        ):  # check for half-filled pixels
            print(
                " Replacing calculated sky area {} with (nominal_area)/2".format(
                    skyarea
                )
            )
            skyarea = nominal_skyarea / 2.0

    return skyarea


def write_output_mock_to_disk(
    output_color_mock_fname,
    output_mock,
    commit_hash,
    seed,
    synthetic_params,
    cutout_number,
    Nside,
    cosmology,
    versionMajor=1,
    versionMinor=1,
    versionMinorMinor=1,
):
    """
    Write the assembled mock to specified output file in hdf5 format
    """

    print(
        "\n...Writing to file {} using commit hash {}".format(
            output_color_mock_fname, commit_hash
        )
    )
    hdfFile = h5py.File(output_color_mock_fname, "w")
    hdfFile.create_group("metaData")
    hdfFile["metaData"]["commit_hash"] = commit_hash
    hdfFile["metaData"]["seed"] = seed
    hdfFile["metaData"]["versionMajor"] = versionMajor
    hdfFile["metaData"]["versionMinor"] = versionMinor
    hdfFile["metaData"]["versionMinorMinor"] = versionMinorMinor
    hdfFile["metaData"]["H_0"] = cosmology.H0.value
    hdfFile["metaData"]["Omega_matter"] = cosmology.Om0
    hdfFile["metaData"]["Omega_b"] = cosmology.Ob0
    hdfFile["metaData"]["skyArea"] = get_skyarea(output_mock, Nside)
    hdfFile["metaData"]["healpix_cutout_number"] = cutout_number
    if synthetic_params and not synthetic_params["skip_synthetics"]:
        synthetic_halo_minimum_mass = synthetic_params["synthetic_halo_minimum_mass"]
        hdfFile["metaData"]["synthetic_halo_minimum_mass"] = synthetic_halo_minimum_mass
    # save software versions
    versions = infer_software_versions()
    for k, v in versions.items():
        hdfFile["metaData"][k] = v

    for k, v in output_mock.items():
        gGroup = hdfFile.create_group(k)
        check_time = time()
        for tk in v.keys():
            # gGroup[tk] = v[tk].quantity.value
            gGroup[tk] = v[tk]

        print(
            ".....time to write group {} = {:.4f} secs".format(k, time() - check_time)
        )

    check_time = time()
    hdfFile.close()
    print(".....time to close file {:.4f} secs".format(time() - check_time))
    print(".....time to close file {:.4f} secs".format(time() - check_time))
