"""
"""
import numpy as np
from diffsky.experimental.dspspop.boris_dust import (
    DEFAULT_U_PARAMS as DEFAULT_FUNO_U_PARAMS,
)
from diffsky.experimental.dspspop.burstshapepop import DEFAULT_BURSTSHAPE_U_PARAMS
from diffsky.experimental.dspspop.dust_deltapop import DEFAULT_DUST_DELTA_U_PARAMS
from diffsky.experimental.dspspop.lgavpop import DEFAULT_LGAV_U_PARAMS
from diffsky.experimental.dspspop.lgfburstpop import DEFAULT_LGFBURST_U_PARAMS

from ..photometry_lc_interp import get_diffsky_sed_info


def test_get_diffsky_sed_info():
    n_met, n_age = 12, 40

    ssp_lgmet = np.linspace(-3, -1, n_met)
    ssp_lg_age_gyr = np.linspace(5, 10.25, n_age) - 9.0

    n_t = 100
    gal_t_table = np.linspace(0.1, 13.8, n_t)

    n_gals = 150
    gal_z_obs = np.random.uniform(0.01, 2.5, n_gals)

    gal_sfr_table = np.random.uniform(0, 100, n_gals * n_t).reshape((n_gals, n_t))

    Om0, w0, wa, h = 0.3, -1, 0.0, 0.7
    cosmo_params = np.array((Om0, w0, wa, h))

    n_wave_seds = 300
    ssp_rest_seds = np.random.uniform(size=(n_met, n_age, n_wave_seds))

    n_rest_filters, n_obs_filters = 2, 3
    n_trans_wave = 40
    obs_filter_waves = np.tile(
        np.linspace(100, 5_000, n_trans_wave), n_obs_filters
    ).reshape((n_obs_filters, n_trans_wave))
    obs_filter_trans = np.ones_like(obs_filter_waves)

    rest_filter_waves = np.tile(
        np.linspace(100, 5_000, n_trans_wave), n_rest_filters
    ).reshape((n_rest_filters, n_trans_wave))
    rest_filter_trans = np.ones_like(rest_filter_waves)

    n_z_table = 23
    ssp_z_table = np.linspace(0.001, 10, n_z_table)
    ssp_restmag_table = np.random.uniform(size=(n_met, n_age, n_rest_filters))
    ssp_obsmag_table = np.random.uniform(size=(n_z_table, n_met, n_age, n_obs_filters))

    _res = get_diffsky_sed_info(
        ssp_z_table,
        ssp_rest_seds,
        ssp_restmag_table,
        ssp_obsmag_table,
        ssp_lgmet,
        ssp_lg_age_gyr,
        gal_t_table,
        gal_z_obs,
        gal_sfr_table,
        cosmo_params,
        rest_filter_waves,
        rest_filter_trans,
        obs_filter_waves,
        obs_filter_trans,
        DEFAULT_LGFBURST_U_PARAMS,
        DEFAULT_BURSTSHAPE_U_PARAMS,
        DEFAULT_LGAV_U_PARAMS,
        DEFAULT_DUST_DELTA_U_PARAMS,
        DEFAULT_FUNO_U_PARAMS,
    )
    for x in _res:
        assert np.all(np.isfinite(x))

    (
        weights,
        gal_frac_trans_obs,
        gal_frac_trans_rest,
        gal_att_curve_params,
        gal_frac_unobs,
        gal_rest_seds,
        gal_obsmags_nodust,
        gal_restmags_nodust,
        gal_obsmags_dust,
        gal_restmags_dust,
    ) = _res
    assert weights.shape == (n_gals, n_met, n_age, 1)
    assert gal_frac_trans_obs.shape == (n_gals, n_age, n_obs_filters)
    assert gal_frac_trans_rest.shape == (n_gals, n_age, n_rest_filters)
    assert gal_att_curve_params.shape == (n_gals, 3)
    assert gal_frac_unobs.shape == (n_gals, n_age)

    assert gal_rest_seds.shape == (n_gals, n_wave_seds)

    assert gal_obsmags_nodust.shape == (n_gals, n_obs_filters)
    assert gal_restmags_nodust.shape == (n_gals, n_rest_filters)
    assert gal_obsmags_dust.shape == (n_gals, n_obs_filters)
    assert gal_restmags_dust.shape == (n_gals, n_rest_filters)

    assert np.all(gal_obsmags_dust >= gal_obsmags_nodust)
    assert np.any(gal_obsmags_dust > gal_obsmags_nodust)

    assert np.all(gal_restmags_dust >= gal_restmags_nodust)
    assert np.any(gal_restmags_dust > gal_restmags_nodust)
