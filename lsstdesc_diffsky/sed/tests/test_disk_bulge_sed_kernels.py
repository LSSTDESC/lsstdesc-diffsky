"""
"""
import numpy as np
from dsps.data_loaders.retrieve_fake_fsps_data import load_fake_ssp_data

from ... import read_mock_params
from ...defaults import DEFAULT_DIFFGAL_PARAMS, DEFAULT_FBULGE_PARAMS
from ..disk_bulge_sed_kernels import calc_rest_sed_disk_bulge_knot_singlegal


def test_calc_rest_sed_evaluates_with_roman_rubin_2023_params():
    all_params = read_mock_params("roman_rubin_2023")

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
        ssp_data.ssp_lgmet,
        ssp_data.ssp_lg_age_gyr,
        ssp_data.ssp_wave,
        ssp_data.ssp_flux,
        *all_params,
    )
    for x in _res:
        assert np.all(np.isfinite(x))
