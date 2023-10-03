"""
"""
import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from diffstar.defaults import DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS
from dsps.data_loaders.retrieve_fake_fsps_data import load_fake_ssp_data
from dsps.metallicity.defaults import DEFAULT_MET_PARAMS

from ..sed_kernels import calc_rest_sed_singlegal


def test_calc_rest_sed_evaluates():
    z_obs = 0.1

    ssp_data = load_fake_ssp_data()
    _res = calc_rest_sed_singlegal(
        z_obs,
        DEFAULT_MAH_PARAMS,
        DEFAULT_MS_PARAMS,
        DEFAULT_Q_PARAMS,
        DEFAULT_MET_PARAMS,
        ssp_data.ssp_lgmet,
        ssp_data.ssp_lg_age_gyr,
        ssp_data.ssp_flux,
    )
    logsm_t_obs, lgmet_t_obs, lgmet_weights, smooth_age_weights = _res
    for x in _res:
        assert np.all(np.isfinite(x))

    assert np.allclose(np.sum(lgmet_weights), 1.0, rtol=1e-3)
    assert np.allclose(np.sum(smooth_age_weights), 1.0, rtol=1e-3)
