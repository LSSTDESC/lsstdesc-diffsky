"""
"""
import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from diffstar.defaults import DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS
from dsps.data_loaders.retrieve_fake_fsps_data import load_fake_ssp_data

from ... import read_diffskypop_params
from ..sed_kernels import calc_rest_sed_galpop, calc_rest_sed_singlegal


def test_calc_rest_sed_evaluates_with_default_params():
    z_obs = 0.1

    ssp_data = load_fake_ssp_data()
    diffskypop_params = read_diffskypop_params("roman_rubin_2023")
    _res = calc_rest_sed_singlegal(
        z_obs,
        DEFAULT_MAH_PARAMS,
        DEFAULT_MS_PARAMS,
        DEFAULT_Q_PARAMS,
        ssp_data,
        diffskypop_params,
    )
    (
        rest_sed,
        rest_sed_nodust,
        logsm_t_obs,
        lgmet_t_obs,
    ) = _res
    for x in _res:
        assert np.all(np.isfinite(x))

    n_met, n_age, n_wave = ssp_data.ssp_flux.shape

    assert np.all(rest_sed <= rest_sed_nodust)
    assert np.any(rest_sed < rest_sed_nodust)

    assert rest_sed.shape == rest_sed_nodust.shape
    assert rest_sed.shape == (n_wave,)


def test_calc_rest_sed_evaluates_with_roman_rubin_2023_params():
    all_params = read_diffskypop_params("roman_rubin_2023")

    z_obs = 0.1

    ssp_data = load_fake_ssp_data()
    _res = calc_rest_sed_singlegal(
        z_obs,
        DEFAULT_MAH_PARAMS,
        DEFAULT_MS_PARAMS,
        DEFAULT_Q_PARAMS,
        ssp_data,
        all_params,
    )
    (
        rest_sed,
        rest_sed_nodust,
        logsm_t_obs,
        lgmet_t_obs,
    ) = _res
    for x in _res:
        assert np.all(np.isfinite(x))

    n_met, n_age, n_wave = ssp_data.ssp_flux.shape

    assert np.all(rest_sed <= rest_sed_nodust)
    assert np.any(rest_sed < rest_sed_nodust)

    assert rest_sed.shape == rest_sed_nodust.shape
    assert rest_sed.shape == (n_wave,)


def test_calc_rest_sed_galpop():
    n_gals = 5

    z_obs_galpop = np.random.uniform(0, 1, n_gals)

    mah_params_galpop = np.tile(DEFAULT_MAH_PARAMS, n_gals)
    mah_params_galpop = mah_params_galpop.reshape((n_gals, -1))

    ms_params_galpop = np.tile(DEFAULT_MS_PARAMS, n_gals)
    ms_params_galpop = ms_params_galpop.reshape((n_gals, -1))

    q_params_galpop = np.tile(DEFAULT_Q_PARAMS, n_gals)
    q_params_galpop = q_params_galpop.reshape((n_gals, -1))

    ssp_data = load_fake_ssp_data()
    n_met, n_age, n_wave = ssp_data.ssp_flux.shape

    all_mock_params = read_diffskypop_params("roman_rubin_2023")

    _res = calc_rest_sed_galpop(
        z_obs_galpop,
        mah_params_galpop,
        ms_params_galpop,
        q_params_galpop,
        ssp_data,
        all_mock_params,
    )
    for x in _res:
        assert np.all(np.isfinite(x))

    rest_sed_galpop, rest_sed_nodust_galpop = _res[:2]
    logsm_t_obs_galpop, lgmet_t_obs_galpop = _res[2:]
    assert rest_sed_galpop.shape == (n_gals, n_wave)

    assert np.all(rest_sed_galpop <= rest_sed_nodust_galpop)
    assert np.any(rest_sed_galpop < rest_sed_nodust_galpop)
    assert np.all(logsm_t_obs_galpop > 0)
    assert np.all(logsm_t_obs_galpop < 15)

    assert np.all(lgmet_t_obs_galpop > -4)
    assert np.all(lgmet_t_obs_galpop < 1)
