"""
"""
# flake8: noqa

import numpy as np
from dsps.cosmology.flat_wcdm import age_at_z0
from dsps.data_loaders.retrieve_fake_fsps_data import (
    load_fake_filter_transmission_curves,
    load_fake_ssp_data,
)
from jax import random as jran

from ... import read_diffskypop_params
from ...defaults import DEFAULT_DIFFGAL_PARAMS, OUTER_RIM_COSMO_PARAMS
from ...legacy.roman_rubin_2023.dsps.data_loaders.retrieve_fake_fsps_data import (
    load_fake_ssp_data_singlemet,
)
from ..photometry_kernels_singlemet import calc_photometry_galpop
from ..photometry_lc_interp_singlemet import get_diffsky_sed_info_singlemet
from ..precompute_ssp_tables import (
    precompute_ssp_obsmags_on_z_table,
    precompute_ssp_obsmags_on_z_table_singlemet,
    precompute_ssp_restmags,
    precompute_ssp_restmags_singlemet,
)


def test_precompute_and_exact_photometry_agree():
    n_gals = 3

    ran_key = jran.PRNGKey(0)
    z_obs_key, morphology_key = jran.split(ran_key, 2)
    z_obs_galpop = jran.uniform(z_obs_key, minval=0.02, maxval=1, shape=(n_gals,))

    DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS = DEFAULT_DIFFGAL_PARAMS

    mah_params_galpop = np.tile(DEFAULT_MAH_PARAMS, n_gals)
    mah_params_galpop = mah_params_galpop.reshape((n_gals, -1))

    ms_params_galpop = np.tile(DEFAULT_MS_PARAMS, n_gals)
    ms_params_galpop = ms_params_galpop.reshape((n_gals, -1))

    q_params_galpop = np.tile(DEFAULT_Q_PARAMS, n_gals)
    q_params_galpop = q_params_galpop.reshape((n_gals, -1))

    ssp_data = load_fake_ssp_data_singlemet()
    n_age, n_wave = ssp_data.ssp_flux.shape

    diffskypop_params = read_diffskypop_params("roman_rubin_2023")

    wave, u, g, r, i, z, y = load_fake_filter_transmission_curves()
    rest_filter_waves = np.tile(wave, 2).reshape(2, (wave.size))
    obs_filter_waves = np.tile(wave, 2).reshape(2, (wave.size))
    rest_filter_trans = np.array((u, g))
    obs_filter_trans = np.array((u, g))
    n_obs_filters = rest_filter_waves.shape[0]
    n_rest_filters = obs_filter_waves.shape[0]

    args = (
        z_obs_galpop,
        mah_params_galpop,
        ms_params_galpop,
        q_params_galpop,
        ssp_data,
        diffskypop_params,
        rest_filter_waves,
        rest_filter_trans,
        obs_filter_waves,
        obs_filter_trans,
        OUTER_RIM_COSMO_PARAMS,
    )

    _res = calc_photometry_galpop(*args)
    rest_mags, obs_mags, rest_mags_nodust, obs_mags_nodust = _res

    ssp_z_table = np.linspace(z_obs_galpop.min() / 2, z_obs_galpop.max() + 0.1, 51)

    ssp_restmag_table = precompute_ssp_restmags_singlemet(
        ssp_data.ssp_wave, ssp_data.ssp_flux, rest_filter_waves, rest_filter_trans
    )
    ssp_obsmag_table = precompute_ssp_obsmags_on_z_table_singlemet(
        ssp_data.ssp_wave,
        ssp_data.ssp_flux,
        obs_filter_waves,
        obs_filter_trans,
        ssp_z_table,
        *OUTER_RIM_COSMO_PARAMS[:-1],
    )
    t0 = age_at_z0(*OUTER_RIM_COSMO_PARAMS[:-1])

    gal_t_table = np.linspace(0.1, t0, 50)

    args = (
        morphology_key,
        z_obs_galpop,
        mah_params_galpop,
        ms_params_galpop,
        q_params_galpop,
        ssp_z_table,
        ssp_restmag_table,
        ssp_obsmag_table,
        ssp_data,
        gal_t_table,
        rest_filter_waves,
        rest_filter_trans,
        obs_filter_waves,
        obs_filter_trans,
        diffskypop_params,
        OUTER_RIM_COSMO_PARAMS,
    )
    sed_info = get_diffsky_sed_info_singlemet(*args)

    atol = 0.1
    assert np.allclose(sed_info.gal_obsmags_dust, obs_mags, atol=atol)
    assert np.allclose(sed_info.gal_restmags_dust, rest_mags, atol=atol)
    assert np.allclose(sed_info.gal_obsmags_nodust, obs_mags_nodust, atol=atol)
    assert np.allclose(sed_info.gal_restmags_nodust, rest_mags_nodust, atol=atol)

    assert obs_mags.shape == (n_gals, n_obs_filters)
    assert rest_mags.shape == (n_gals, n_rest_filters)
    assert obs_mags_nodust.shape == (n_gals, n_obs_filters)
    assert rest_mags_nodust.shape == (n_gals, n_rest_filters)


def test_precompute_photometry_correctly_handles_fb():
    n_gals = 3

    ran_key = jran.PRNGKey(0)
    z_obs_key, morphology_key = jran.split(ran_key, 2)
    z_obs_galpop = jran.uniform(z_obs_key, minval=0.02, maxval=1, shape=(n_gals,))

    DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS = DEFAULT_DIFFGAL_PARAMS

    mah_params_galpop = np.tile(DEFAULT_MAH_PARAMS, n_gals)
    mah_params_galpop = mah_params_galpop.reshape((n_gals, -1))

    ms_params_galpop = np.tile(DEFAULT_MS_PARAMS, n_gals)
    ms_params_galpop = ms_params_galpop.reshape((n_gals, -1))

    q_params_galpop = np.tile(DEFAULT_Q_PARAMS, n_gals)
    q_params_galpop = q_params_galpop.reshape((n_gals, -1))

    ssp_data = load_fake_ssp_data_singlemet()

    diffskypop_params = read_diffskypop_params("roman_rubin_2023")

    wave, u, g, r, i, z, y = load_fake_filter_transmission_curves()
    rest_filter_waves = np.tile(wave, 2).reshape(2, (wave.size))
    obs_filter_waves = np.tile(wave, 2).reshape(2, (wave.size))
    rest_filter_trans = np.array((u, g))
    obs_filter_trans = np.array((u, g))

    ssp_z_table = np.linspace(z_obs_galpop.min() / 2, z_obs_galpop.max() + 0.1, 51)

    ssp_restmag_table = precompute_ssp_restmags_singlemet(
        ssp_data.ssp_wave, ssp_data.ssp_flux, rest_filter_waves, rest_filter_trans
    )
    ssp_obsmag_table = precompute_ssp_obsmags_on_z_table_singlemet(
        ssp_data.ssp_wave,
        ssp_data.ssp_flux,
        obs_filter_waves,
        obs_filter_trans,
        ssp_z_table,
        *OUTER_RIM_COSMO_PARAMS[:-1],
    )
    t0 = age_at_z0(*OUTER_RIM_COSMO_PARAMS[:-1])

    gal_t_table = np.linspace(0.1, t0, 50)

    args = (
        morphology_key,
        z_obs_galpop,
        mah_params_galpop,
        ms_params_galpop,
        q_params_galpop,
        ssp_z_table,
        ssp_restmag_table,
        ssp_obsmag_table,
        ssp_data,
        gal_t_table,
        rest_filter_waves,
        rest_filter_trans,
        obs_filter_waves,
        obs_filter_trans,
        diffskypop_params,
        OUTER_RIM_COSMO_PARAMS,
    )
    sed_info = get_diffsky_sed_info_singlemet(*args)

    cosmo_pars2 = (*OUTER_RIM_COSMO_PARAMS[:-1], 0.1)
    args = (
        morphology_key,
        z_obs_galpop,
        mah_params_galpop,
        ms_params_galpop,
        q_params_galpop,
        ssp_z_table,
        ssp_restmag_table,
        ssp_obsmag_table,
        ssp_data,
        gal_t_table,
        rest_filter_waves,
        rest_filter_trans,
        obs_filter_waves,
        obs_filter_trans,
        diffskypop_params,
        cosmo_pars2,
    )
    sed_info2 = get_diffsky_sed_info_singlemet(*args)

    # fb2 < fb so there galaxies should be fainter in cosmo2
    assert np.all(sed_info.gal_restmags_dust < sed_info2.gal_restmags_dust)


def test_precompute_ssp_obsmags_agrees_with_and_without_metallicity_dimension():
    n_gals = 3

    ran_key = jran.PRNGKey(0)
    z_obs_key, morphology_key = jran.split(ran_key, 2)
    z_obs_galpop = jran.uniform(z_obs_key, minval=0.02, maxval=1, shape=(n_gals,))

    DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS = DEFAULT_DIFFGAL_PARAMS

    mah_params_galpop = np.tile(DEFAULT_MAH_PARAMS, n_gals)
    mah_params_galpop = mah_params_galpop.reshape((n_gals, -1))

    ms_params_galpop = np.tile(DEFAULT_MS_PARAMS, n_gals)
    ms_params_galpop = ms_params_galpop.reshape((n_gals, -1))

    q_params_galpop = np.tile(DEFAULT_Q_PARAMS, n_gals)
    q_params_galpop = q_params_galpop.reshape((n_gals, -1))

    ssp_data = load_fake_ssp_data()
    n_met, n_age, n_wave = ssp_data.ssp_flux.shape
    ssp_data_singlemet = load_fake_ssp_data_singlemet()
    ssp_data.ssp_flux[:, :, :] = ssp_data_singlemet.ssp_flux

    wave, u, g = load_fake_filter_transmission_curves()[:3]
    obs_filter_waves = np.tile(wave, 2).reshape(2, (wave.size))
    obs_filter_trans = np.array((u, g))
    n_obs_filters = obs_filter_waves.shape[0]

    n_z_table = 51
    ssp_z_table = np.linspace(
        z_obs_galpop.min() / 2, z_obs_galpop.max() + 0.1, n_z_table
    )

    ssp_obsmag_table = precompute_ssp_obsmags_on_z_table(
        ssp_data.ssp_wave,
        ssp_data.ssp_flux,
        obs_filter_waves,
        obs_filter_trans,
        ssp_z_table,
        *OUTER_RIM_COSMO_PARAMS[:-1],
    )
    assert ssp_obsmag_table.shape == (n_z_table, n_met, n_age, n_obs_filters)

    ssp_obsmag_table_singlemet = precompute_ssp_obsmags_on_z_table_singlemet(
        ssp_data.ssp_wave,
        ssp_data_singlemet.ssp_flux,
        obs_filter_waves,
        obs_filter_trans,
        ssp_z_table,
        *OUTER_RIM_COSMO_PARAMS[:-1],
    )
    assert ssp_obsmag_table_singlemet.shape == (n_z_table, n_age, n_obs_filters)

    for iz in range(n_met):
        assert np.allclose(ssp_obsmag_table[:, iz, :, :], ssp_obsmag_table_singlemet)


def test_precompute_ssp_restmags_agrees_with_and_without_metallicity_dimension():
    n_gals = 3

    ran_key = jran.PRNGKey(0)
    z_obs_key, morphology_key = jran.split(ran_key, 2)
    z_obs_galpop = jran.uniform(z_obs_key, minval=0.02, maxval=1, shape=(n_gals,))

    DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS = DEFAULT_DIFFGAL_PARAMS

    mah_params_galpop = np.tile(DEFAULT_MAH_PARAMS, n_gals)
    mah_params_galpop = mah_params_galpop.reshape((n_gals, -1))

    ms_params_galpop = np.tile(DEFAULT_MS_PARAMS, n_gals)
    ms_params_galpop = ms_params_galpop.reshape((n_gals, -1))

    q_params_galpop = np.tile(DEFAULT_Q_PARAMS, n_gals)
    q_params_galpop = q_params_galpop.reshape((n_gals, -1))

    ssp_data = load_fake_ssp_data()
    n_met, n_age, n_wave = ssp_data.ssp_flux.shape
    ssp_data_singlemet = load_fake_ssp_data_singlemet()
    ssp_data.ssp_flux[:, :, :] = ssp_data_singlemet.ssp_flux

    wave, u, g = load_fake_filter_transmission_curves()[:3]
    rest_filter_waves = np.tile(wave, 2).reshape(2, (wave.size))
    rest_filter_trans = np.array((u, g))
    n_rest_filters = rest_filter_waves.shape[0]

    ssp_restmag_table = precompute_ssp_restmags(
        ssp_data.ssp_wave, ssp_data.ssp_flux, rest_filter_waves, rest_filter_trans
    )

    ssp_restmag_table_singlemet = precompute_ssp_restmags_singlemet(
        ssp_data.ssp_wave,
        ssp_data_singlemet.ssp_flux,
        rest_filter_waves,
        rest_filter_trans,
    )
    assert ssp_restmag_table_singlemet.shape == (n_age, n_rest_filters)

    for iz in range(n_met):
        assert np.allclose(ssp_restmag_table[iz, :, :], ssp_restmag_table_singlemet)
