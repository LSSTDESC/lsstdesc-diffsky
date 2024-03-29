"""
"""
import os

import numpy as np

from ... import read_diffskypop_params
from ...defaults import DEFAULT_DIFFGAL_PARAMS, DEFAULT_FBULGE_PARAMS
from ...legacy.roman_rubin_2023.dsps.data_loaders.retrieve_fake_fsps_data import (
    load_fake_ssp_data_singlemet,
)
from ..disk_bulge_sed_kernels_singlemet import (
    calc_rest_sed_disk_bulge_knot_galpop,
    calc_rest_sed_disk_bulge_knot_singlegal,
)

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))
TESTING_DATA_DRN = os.path.join(_THIS_DRNAME, "testing_data")


def test_calc_rest_sed_evaluates_with_roman_rubin_2023_params():
    diffskypop_params = read_diffskypop_params("roman_rubin_2023")

    z_obs = 0.1

    ssp_data = load_fake_ssp_data_singlemet()
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
        diffskypop_params,
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

    ssp_data = load_fake_ssp_data_singlemet()
    n_age, n_wave = ssp_data.ssp_flux.shape

    diffskypop_params = read_diffskypop_params("roman_rubin_2023")

    _res = calc_rest_sed_disk_bulge_knot_galpop(
        z_obs_galpop,
        mah_params_galpop,
        ms_params_galpop,
        q_params_galpop,
        fbulge_params_galpop,
        fknot_galpop,
        ssp_data,
        diffskypop_params,
    )
    for x in _res:
        assert np.all(np.isfinite(x))


def test_calc_rest_sed_has_frozen_behavior_on_defaults():
    diffskypop_params = read_diffskypop_params("roman_rubin_2023")

    z_obs = 0.1

    ssp_data = load_fake_ssp_data_singlemet()
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
        diffskypop_params,
    )
    rest_sed_bulge, rest_sed_dd, rest_sed_knot = _res[:3]

    fn = os.path.join(TESTING_DATA_DRN, "rest_sed_bulge_singlemet_default_params.txt")
    rest_sed_bulge_frozen = np.loadtxt(fn)
    assert np.allclose(rest_sed_bulge, rest_sed_bulge_frozen, rtol=1e-4)

    fn = os.path.join(TESTING_DATA_DRN, "rest_sed_dd_singlemet_default_params.txt")
    rest_sed_dd_frozen = np.loadtxt(fn)
    assert np.allclose(rest_sed_dd, rest_sed_dd_frozen, rtol=1e-4)

    fn = os.path.join(TESTING_DATA_DRN, "rest_sed_knot_singlemet_default_params.txt")
    rest_sed_knot_frozen = np.loadtxt(fn)
    assert np.allclose(rest_sed_knot, rest_sed_knot_frozen, rtol=1e-4)
