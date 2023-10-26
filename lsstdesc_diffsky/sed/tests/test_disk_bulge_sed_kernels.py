"""
"""
import numpy as np
from dsps.data_loaders.retrieve_fake_fsps_data import load_fake_ssp_data

from ... import read_diffskypop_params
from ...defaults import DEFAULT_DIFFGAL_PARAMS, DEFAULT_FBULGE_PARAMS
from ..disk_bulge_sed_kernels import (
    calc_rest_sed_disk_bulge_knot_galpop,
    calc_rest_sed_disk_bulge_knot_singlegal,
)


def test_calc_rest_sed_evaluates_with_roman_rubin_2023_params():
    all_params = read_diffskypop_params("roman_rubin_2023")

    z_obs = 0.1

    ssp_data = load_fake_ssp_data()
    mah_params, ms_params, q_params = DEFAULT_DIFFGAL_PARAMS

    fknot = 0.05
    _res = calc_rest_sed_disk_bulge_knot_singlegal(
        z_obs,
        mah_params,
        ms_params,
        q_params,
        DEFAULT_FBULGE_PARAMS,
        fknot,
        ssp_data,
        all_params,
    )
    for x in _res:
        assert np.all(np.isfinite(x))


def test_calc_rest_sed_disk_bulge_knot_galpop():
    n_gals = 5

    z_obs_galpop = np.random.uniform(0.02, 1, n_gals)

    mah_params, ms_params, q_params = DEFAULT_DIFFGAL_PARAMS

    mah_params_galpop = np.tile(mah_params, n_gals)
    mah_params_galpop = mah_params_galpop.reshape((n_gals, -1))

    ms_params_galpop = np.tile(ms_params, n_gals)
    ms_params_galpop = ms_params_galpop.reshape((n_gals, -1))

    q_params_galpop = np.tile(q_params, n_gals)
    q_params_galpop = q_params_galpop.reshape((n_gals, -1))

    fbulge_params_galpop = np.tile(DEFAULT_FBULGE_PARAMS, n_gals)
    fbulge_params_galpop = fbulge_params_galpop.reshape((n_gals, -1))

    fknot_galpop = np.random.uniform(0, 0.1, n_gals)

    ssp_data = load_fake_ssp_data()
    n_met, n_age, n_wave = ssp_data.ssp_flux.shape

    all_mock_params = read_diffskypop_params("roman_rubin_2023")

    _res = calc_rest_sed_disk_bulge_knot_galpop(
        z_obs_galpop,
        mah_params_galpop,
        ms_params_galpop,
        q_params_galpop,
        fbulge_params_galpop,
        fknot_galpop,
        ssp_data,
        all_mock_params,
    )
    for x in _res:
        assert np.all(np.isfinite(x))
