"""
"""
import os

import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from diffstar.defaults import DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS

from ... import read_diffskypop_params
from ...legacy.roman_rubin_2023.dsps.data_loaders.retrieve_fake_fsps_data import (
    load_fake_ssp_data_singlemet,
)
from ..sed_kernels_singlemet import calc_rest_sed_galpop, calc_rest_sed_singlegal

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TESTING_DATA_DRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_calc_rest_sed_evaluates_with_default_params():
    z_obs = 0.1
    ssp_data = load_fake_ssp_data_singlemet()
    diffskypop_data = read_diffskypop_params("roman_rubin_2023")
    _res = calc_rest_sed_singlegal(
        z_obs,
        DEFAULT_MAH_PARAMS,
        DEFAULT_MS_PARAMS,
        DEFAULT_Q_PARAMS,
        ssp_data,
        diffskypop_data,
    )
    (
        rest_sed,
        rest_sed_nodust,
        logsm_t_obs,
    ) = _res
    for x in _res:
        assert np.all(np.isfinite(x))

    n_age, n_wave = ssp_data.ssp_flux.shape

    assert np.all(rest_sed <= rest_sed_nodust)
    assert np.any(rest_sed < rest_sed_nodust)

    assert rest_sed.shape == rest_sed_nodust.shape
    assert rest_sed.shape == (n_wave,)


def test_calc_rest_sed_evaluates_with_roman_rubin_2023_params():
    all_params = read_diffskypop_params("roman_rubin_2023")

    z_obs = 0.1

    ssp_data = load_fake_ssp_data_singlemet()
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
    ) = _res
    for x in _res:
        assert np.all(np.isfinite(x))

    n_age, n_wave = ssp_data.ssp_flux.shape

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

    ssp_data = load_fake_ssp_data_singlemet()
    n_age, n_wave = ssp_data.ssp_flux.shape

    diffskypop_data = read_diffskypop_params("roman_rubin_2023")

    _res = calc_rest_sed_galpop(
        z_obs_galpop,
        mah_params_galpop,
        ms_params_galpop,
        q_params_galpop,
        ssp_data,
        diffskypop_data,
    )
    for x in _res:
        assert np.all(np.isfinite(x))

    rest_sed_galpop, rest_sed_nodust_galpop, logsm_t_obs_galpop = _res[:3]
    assert rest_sed_galpop.shape == (n_gals, n_wave)

    assert np.all(rest_sed_galpop <= rest_sed_nodust_galpop)
    assert np.any(rest_sed_galpop < rest_sed_nodust_galpop)
    assert np.all(logsm_t_obs_galpop > 0)
    assert np.all(logsm_t_obs_galpop < 15)


def test_calc_rest_sed_has_frozen_behavior_on_default_params():
    z_obs = 0.1
    ssp_data = load_fake_ssp_data_singlemet()
    diffskypop_data = read_diffskypop_params("roman_rubin_2023")
    rest_sed = calc_rest_sed_singlegal(
        z_obs,
        DEFAULT_MAH_PARAMS,
        DEFAULT_MS_PARAMS,
        DEFAULT_Q_PARAMS,
        ssp_data,
        diffskypop_data,
    )[0]

    fn = os.path.join(TESTING_DATA_DRN, "rest_sed_singlemet_default_params.txt")
    rest_sed_frozen = np.loadtxt(fn)
    assert np.allclose(rest_sed, rest_sed_frozen, rtol=1e-4)
