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
from diffstar.fitting_helpers.stars import _integrate_sfr
from dsps.experimental.diffburst import DEFAULT_PARAMS as DEFAULT_BURST_PARAMS
from dsps.experimental.diffburst import _age_weights_from_params
from dsps.utils import _jax_get_dt_array
from jax import jit as jjit
from jax import random as jran
from jax import vmap

_B = (0, None)
_integrate_sfr_vmap = jjit(vmap(_integrate_sfr, in_axes=_B))

from ..photometry_lc_interp import (
    _linterp_vmap,
    decompose_sfh_into_bulge_disk_knots,
    get_diffsky_sed_info,
)


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


def test_decompose_sfh_into_bulge_disk_knots():
    ran_key = jran.PRNGKey(0)

    n_age = 40
    ssp_lg_age_yr = np.linspace(5, 10.25, n_age)
    ssp_lg_age_gyr = ssp_lg_age_yr - 9.0

    n_t = 100
    tmin, tmax = 0.1, 13.8
    gal_t_table = np.linspace(tmin, tmax, n_t)

    n_gals = 150

    gal_t_obs = np.random.uniform(tmin, tmax, n_gals)
    gal_sfh = np.random.uniform(0, 100, n_gals * n_t).reshape((n_gals, n_t))
    fburst = 10 ** np.random.uniform(-4, -2, n_gals)

    age_weights_singleburst = _age_weights_from_params(
        ssp_lg_age_yr, DEFAULT_BURST_PARAMS
    )
    age_weights_burstpop = np.tile(age_weights_singleburst, n_gals)
    age_weights_burstpop = age_weights_burstpop.reshape((n_gals, n_age))

    args = (
        ran_key,
        gal_t_obs,
        gal_t_table,
        gal_sfh,
        fburst,
        age_weights_burstpop,
        ssp_lg_age_gyr,
    )
    _res = decompose_sfh_into_bulge_disk_knots(*args)

    mbulge, mdd, mknot, mburst = _res[:4]
    bulge_age_weights, dd_age_weights, knot_age_weights = _res[4:]

    mtot = mbulge + mdd + mknot
    lgmtot = np.log10(mtot)

    dt_table = _jax_get_dt_array(gal_t_table)
    gal_smh = np.cumsum(gal_sfh * dt_table, axis=1) * 1e9
    gal_logsmh = np.log10(gal_smh)

    lgt_table = np.log10(gal_t_table)
    lgmstar_t_obs = _linterp_vmap(np.log10(gal_t_obs), lgt_table, gal_logsmh)

    assert np.allclose(lgmstar_t_obs, lgmtot, atol=0.01)
