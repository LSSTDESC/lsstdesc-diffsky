"""
"""
import numpy as np
from diffmah.defaults import DEFAULT_MAH_PARAMS
from diffsky.experimental.dspspop.boris_dust import (
    DEFAULT_U_PARAMS as DEFAULT_FUNO_U_PARAMS,
)
from diffsky.experimental.dspspop.burstshapepop import DEFAULT_BURSTSHAPE_U_PARAMS
from diffsky.experimental.dspspop.dust_deltapop import DEFAULT_DUST_DELTA_U_PARAMS
from diffsky.experimental.dspspop.lgavpop import DEFAULT_LGAV_U_PARAMS
from diffsky.experimental.dspspop.lgfburstpop import DEFAULT_LGFBURST_U_PARAMS
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
        ssp_data.ssp_lgmet,
        ssp_data.ssp_lg_age_gyr,
        ssp_data.ssp_wave,
        ssp_data.ssp_flux,
        DEFAULT_LGFBURST_U_PARAMS,
        DEFAULT_BURSTSHAPE_U_PARAMS,
        DEFAULT_LGAV_U_PARAMS,
        DEFAULT_DUST_DELTA_U_PARAMS,
        DEFAULT_FUNO_U_PARAMS,
        DEFAULT_MET_PARAMS,
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
