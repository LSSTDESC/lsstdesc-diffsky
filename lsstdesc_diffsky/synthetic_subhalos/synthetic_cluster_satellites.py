import numpy as np
from astropy.table import Table
from astropy.utils.misc import NumpyRNGContext
from halotools.empirical_models import NFWPhaseSpace
from ..triaxial_satellite_distributions.monte_carlo_triaxial_profile import (
    generate_triaxial_satellite_distribution,
)

__all__ = ("calculate_synthetic_richness", "model_synthetic_cluster_satellites")


def calculate_synthetic_richness(
    halo_richness,
    logmhalo,
    logmhalo_source,
    cluster_satboost_logm_table,
    cluster_satboost_table,
    logm_outer_rim_effect=14.75,
):
    """
    Parameters
    ----------
    halo_richness : ndarray
        Numpy array of shape (nhalos, ) storing the richness of each halo

    logmhalo : ndarray
        Numpy array of shape (nhalos, ) storing the log mass of each halo

    Returns
    -------
    synthetic_richness : ndarray
        Numpy integer array of shape (nhalos, ) storing the synthetic richness
    """
    boost_factor = np.interp(
        logmhalo, cluster_satboost_logm_table, cluster_satboost_table
    )
    dlogm = logmhalo - logmhalo_source
    outer_rim_boost_factor = 10.0**dlogm
    low, high = logm_outer_rim_effect - 0.25, logm_outer_rim_effect + 0.25
    logm_outer_rim_effect = np.random.uniform(low, high, len(boost_factor))
    highmass_mask = logmhalo > logm_outer_rim_effect
    boost_factor = np.where(
        highmass_mask, outer_rim_boost_factor * boost_factor, boost_factor
    )
    return np.array(halo_richness * boost_factor, dtype=int)


def get_ellipsoidal_positions_and_velocities(sats, host_conc=5.0):
    """
    generate positions and velocities base on ellipsoidal distributions
    """
    print(".......using tri-axial positions for synthetic satellites")
    e_sats = {}
    b_to_a = sats["target_halo_axis_B_length"] / sats["target_halo_axis_A_length"]
    c_to_a = sats["target_halo_axis_C_length"] / sats["target_halo_axis_A_length"]
    e_sats["x"], e_sats["y"], e_sats["z"] = generate_triaxial_satellite_distribution(
        host_conc,
        sats["target_halo_axis_A_x"],
        sats["target_halo_axis_A_y"],
        sats["target_halo_axis_A_z"],
        b_to_a,
        c_to_a,
    )

    # compute velocities based on gaussian draw centered on halo velocity
    e_sats["vx"], e_sats["vy"], e_sats["vz"] = get_satellite_velocities(
        sats["target_halo_vx"],
        sats["target_halo_vy"],
        sats["target_halo_vz"],
        sats["target_halo_mass"],
    )

    return e_sats


def get_satellite_velocities(
    halo_vx,
    halo_vy,
    halo_vz,
    halo_mass,
    seed=43,
    seed_inc=2,
    sigma_v0=100.0,
    logmass_v0=12.0,
    sigma_v1=1000.0,
    logmass_v1=15.0,
    sigma_min=10.0,
):
    # setup linear interpolation on log(halomass)
    w = (sigma_v1 - sigma_v0) / (logmass_v1 - logmass_v0)
    w0 = sigma_v0 - w * logmass_v0

    # setup widths based on halo mass and force minimum value
    widths = w0 + w * np.log10(halo_mass)
    mask = widths < sigma_min
    widths[mask] = sigma_min

    with NumpyRNGContext(seed):
        sat_vx = np.random.normal(halo_vx, widths)
    with NumpyRNGContext(seed + seed_inc):
        sat_vy = np.random.normal(halo_vy, widths)
    with NumpyRNGContext(seed + 2 * seed_inc):
        sat_vz = np.random.normal(halo_vz, widths)

    return sat_vx, sat_vy, sat_vz


def model_synthetic_cluster_satellites(
    mock,
    Lbox=256.0,
    source_halo_mass_key="host_halo_mvir",
    source_halo_id_key="source_halo_id",
    upid_key="upid",
    cluster_satboost_logm_table=[13.5, 13.75, 14],
    cluster_satboost_table=[0.0, 0.15, 0.2],
    SFH_keys=[],
    tri_axial_positions=True,
    host_conc=5.0,
    snapshot=False,
    source_galaxy_tag="source_galaxy",
    **kwargs
):
    """ """
    #  Calculate the mass and richness of every target halo
    host_halo_id, idx, counts = np.unique(
        mock["target_halo_id"], return_counts=True, return_index=True
    )
    host_mass = mock["target_halo_mass"][idx]
    host_redshift = mock["target_halo_redshift"][idx]
    host_x = mock["target_halo_x"][idx]
    host_y = mock["target_halo_y"][idx]
    host_z = mock["target_halo_z"][idx]
    host_vx = mock["target_halo_vx"][idx]
    host_vy = mock["target_halo_vy"][idx]
    host_vz = mock["target_halo_vz"][idx]
    source_halo_mvir = mock[source_halo_mass_key][idx]
    target_halo_id = mock["target_halo_id"][idx]
    target_halo_fof_halo_id = mock["target_halo_fof_halo_id"][idx]
    host_sod_mass = mock["sod_halo_mass"][idx]
    host_sod_radius = mock["sod_halo_radius"][idx]
    host_sod_cdelta = mock["sod_halo_cdelta"][idx]
    host_sod_cdelta_error = mock["sod_halo_cdelta_error"][idx]

    #  Calculate tri-axial properties
    tri_axial_properties = (
        "target_halo_ellipticity",
        "target_halo_prolaticity",
        "target_halo_axis_A_length",
        "target_halo_axis_B_length",
        "target_halo_axis_C_length",
        "target_halo_axis_A_x",
        "target_halo_axis_A_y",
        "target_halo_axis_A_z",
    )
    host_tri_axial_properties = {}
    for t in tri_axial_properties:
        host_tri_axial_properties[t] = mock[t][idx]

    #  Light-cone additions
    if not snapshot:
        target_halo_lightcone_replication = mock["lightcone_replication"][idx]
        target_halo_lightcone_rotation = mock["lightcone_rotation"][idx]

    host_richness = counts

    #  For each target halo, calculate the number of synthetic satellites we need to add
    synthetic_richness = calculate_synthetic_richness(
        host_richness,
        np.log10(host_mass),
        np.log10(source_halo_mvir),
        cluster_satboost_logm_table,
        cluster_satboost_table,
    )

    if np.sum(synthetic_richness) <= 1:
        return Table()
    else:
        sats = Table()

        # For every synthetic galaxy, calculate the mass, redshift, position, and
        # velocity of the host halo
        sats["target_halo_mass"] = np.repeat(host_mass, synthetic_richness)
        sats["target_halo_redshift"] = np.repeat(host_redshift, synthetic_richness)
        sats["target_halo_x"] = np.repeat(host_x, synthetic_richness)
        sats["target_halo_y"] = np.repeat(host_y, synthetic_richness)
        sats["target_halo_z"] = np.repeat(host_z, synthetic_richness)
        sats["target_halo_vx"] = np.repeat(host_vx, synthetic_richness)
        sats["target_halo_vy"] = np.repeat(host_vy, synthetic_richness)
        sats["target_halo_vz"] = np.repeat(host_vz, synthetic_richness)
        sats["target_halo_id"] = np.repeat(target_halo_id, synthetic_richness)
        sats["target_halo_fof_halo_id"] = np.repeat(
            target_halo_fof_halo_id, synthetic_richness
        )
        #  Add sod properties
        sats["sod_halo_mass"] = np.repeat(host_sod_mass, synthetic_richness)
        sats["sod_halo_radius"] = np.repeat(host_sod_radius, synthetic_richness)
        sats["sod_halo_cdelta"] = np.repeat(host_sod_cdelta, synthetic_richness)
        sats["sod_halo_cdelta_error"] = np.repeat(
            host_sod_cdelta_error, synthetic_richness
        )

        #  Add tri-axial properties
        for k, v in host_tri_axial_properties.items():
            sats[k] = np.repeat(v, synthetic_richness)

        #  Light-cone additions
        if not snapshot:
            sats["lightcone_replication"] = np.repeat(
                target_halo_lightcone_replication, synthetic_richness
            )
            sats["lightcone_rotation"] = np.repeat(
                target_halo_lightcone_rotation, synthetic_richness
            )

        sats[upid_key] = sats["target_halo_id"]

        if tri_axial_positions:
            nfw_sats = get_ellipsoidal_positions_and_velocities(
                sats, host_conc=host_conc
            )
        else:
            # Use Halotools to generate halo-centric positions and velocities
            # according to NFW
            nfw = NFWPhaseSpace()
            nfw_sats = nfw.mc_generate_nfw_phase_space_points(
                mass=sats["target_halo_mass"]
            )

        sats["host_dx"] = nfw_sats["x"]
        sats["host_dy"] = nfw_sats["y"]
        sats["host_dz"] = nfw_sats["z"]
        sats["host_dvx"] = nfw_sats["vx"]
        sats["host_dvy"] = nfw_sats["vy"]
        sats["host_dvz"] = nfw_sats["vz"]

        #  Add host-centric pos/vel to target halo pos/vel
        sats["x"] = sats["target_halo_x"] + nfw_sats["x"]
        sats["y"] = sats["target_halo_y"] + nfw_sats["y"]
        sats["z"] = sats["target_halo_z"] + nfw_sats["z"]
        sats["vx"] = sats["target_halo_vx"] + nfw_sats["vx"]
        sats["vy"] = sats["target_halo_vy"] + nfw_sats["vy"]
        sats["vz"] = sats["target_halo_vz"] + nfw_sats["vz"]

        if Lbox > 0.0:  # enforce periodicity
            sats["x"] = np.mod(sats["x"], Lbox)
            sats["y"] = np.mod(sats["y"], Lbox)
            sats["z"] = np.mod(sats["z"], Lbox)

        sats[source_halo_id_key] = -1
        sats[source_halo_mass_key] = sats["target_halo_mass"]
        sats["source_halo_uber_hostid"] = -1

        # add source keys and SFH parameter histories randomly from existing
        # satellites
        keys = [
            source_galaxy_tag + "mp",
            source_galaxy_tag + "vmp",
            source_galaxy_tag + "rvir",
            source_galaxy_tag + "upid",
            source_galaxy_tag + "host_rvir",
            source_galaxy_tag + "has_fit",
            source_galaxy_tag + "nofit_replace",
            source_galaxy_tag + "is_main_branch",
            source_galaxy_tag + "obs_sm",
            source_galaxy_tag + "obs_sfr",
            source_galaxy_tag + "sfr_percentile",
        ] + list(SFH_keys)
        # initialize
        for k in keys:
            if "has_fit" in k or "main_branch" in k or "nofit_replace" in k:
                sats[k] = np.zeros(len(sats["target_halo_id"]), dtype=bool)
            else:
                sats[k] = np.zeros(len(sats["target_halo_id"]))

        # find halos requiring fakes
        nonzeros = synthetic_richness > 0
        print(
            ".......adding SFH information for {} synthetic satellites".format(
                np.count_nonzero(nonzeros)
            )
        )
        print(".........using random resampling of existing satellites")
        for ns, halo_id in zip(synthetic_richness[nonzeros], target_halo_id[nonzeros]):
            # identify existing satellites
            maskm = (mock["target_halo_id"] == halo_id) & (
                mock[source_galaxy_tag + "upid"] != -1
            )
            # randomly select from existing satellites to populate synthetics
            indexes = np.random.choice(np.where(maskm)[0], size=ns)
            masks = sats["target_halo_id"] == halo_id
            assert (
                np.count_nonzero(masks) == ns
            ), "Mismatch in number of expected synthetic staellites"
            for k in keys:
                sats[k][masks] = mock[k][indexes]

        # It is important to insure that `sats` the `dc2` mock have the exact same
        # columns, since these two get combined by a call to `astropy.table.vstack`.
        # Here, we enforce this:
        msg = (
            "The synthetic satellites columns must be the same as the regular mock\n"
            "sats keys = {0}\nmock keys = {1}"
        )
        satkeys = list(sats.keys())
        dc2keys = list(mock.keys())
        if [k for k in dc2keys if k not in satkeys]:
            print(".......extra dc2keys:", [k for k in dc2keys if k not in satkeys])
        if [k for k in satkeys if k not in dc2keys]:
            print(".......extra satkeys:", [k for k in satkeys if k not in dc2keys])
        assert set(satkeys) == set(dc2keys), msg.format(satkeys, dc2keys)

        return sats
