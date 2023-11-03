"""
"""
import numpy as np
from dsps.data_loaders.retrieve_fake_fsps_data import load_fake_ssp_data
from dsps.experimental.diffburst import DLGAGE_MIN, LGAGE_MAX, LGYR_PEAK_MIN
from dsps.metallicity.mzr import DEFAULT_MET_PDICT
from jax import random as jran

from ...defaults import DEFAULT_DIFFGAL_PARAMS
from ...disk_bulge_modeling.disk_knots import FKNOT_MAX
from ...dspspop.boris_dust import DEFAULT_U_PARAMS as DEFAULT_FUNO_U_PARAMS
from ...dspspop.burstshapepop import DEFAULT_BURSTSHAPE_U_PARAMS
from ...dspspop.dust_deltapop import DEFAULT_DUST_DELTA_U_PARAMS
from ...dspspop.lgavpop import DEFAULT_LGAV_U_PARAMS
from ...dspspop.lgfburstpop import DEFAULT_LGFBURST_U_PARAMS
from ...param_data.param_reader import DiffskyPopParams
from ..photometry_lc_interp import get_diffsky_sed_info

DEFAULT_MET_PARAMS = np.array(list(DEFAULT_MET_PDICT.values()))


def test_get_diffsky_sed_info():
    ssp_data = load_fake_ssp_data()
    n_met, n_age = ssp_data.ssp_flux.shape[:2]

    n_t = 100
    gal_t_table = np.linspace(0.1, 13.8, n_t)

    n_gals = 150
    gal_z_obs = np.random.uniform(0.01, 2.5, n_gals)

    mah_params, ms_params, q_params = DEFAULT_DIFFGAL_PARAMS
    mah_params_galpop = np.tile(mah_params, n_gals)
    mah_params_galpop = mah_params_galpop.reshape((n_gals, -1))

    ms_params_galpop = np.tile(ms_params, n_gals)
    ms_params_galpop = ms_params_galpop.reshape((n_gals, -1))

    q_params_galpop = np.tile(q_params, n_gals)
    q_params_galpop = q_params_galpop.reshape((n_gals, -1))

    Om0, w0, wa, h, fb = 0.3, -1, 0.0, 0.7, 0.16
    cosmo_params = np.array((Om0, w0, wa, h, fb))

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

    diffskypop_params = DiffskyPopParams(
        DEFAULT_LGFBURST_U_PARAMS,
        DEFAULT_BURSTSHAPE_U_PARAMS,
        DEFAULT_LGAV_U_PARAMS,
        DEFAULT_DUST_DELTA_U_PARAMS,
        DEFAULT_FUNO_U_PARAMS,
        DEFAULT_MET_PARAMS,
    )
    ran_key = jran.PRNGKey(0)
    _res = get_diffsky_sed_info(
        ran_key,
        gal_z_obs,
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
        cosmo_params,
    )
    for x in _res:
        assert np.all(np.isfinite(x))

    (
        weights,
        gal_frac_trans_obs,
        gal_frac_trans_rest,
        gal_att_curve_params,
        gal_frac_unobs,
        gal_fburst,
        gal_burstshape_params,
        gal_frac_bulge_t_obs,
        gal_fbulge_params,
        gal_fknot,
        gal_obsmags_nodust,
        gal_restmags_nodust,
        gal_obsmags_dust,
        gal_restmags_dust,
    ) = _res
    assert weights.shape == (n_gals, n_met, n_age)
    assert gal_frac_trans_obs.shape == (n_gals, n_age, n_obs_filters)
    assert gal_frac_trans_rest.shape == (n_gals, n_age, n_rest_filters)
    assert gal_att_curve_params.shape == (n_gals, 3)
    assert gal_frac_unobs.shape == (n_gals, n_age)

    assert gal_fburst.shape == (n_gals,)
    assert gal_burstshape_params.shape == (n_gals, 2)
    assert np.all(gal_fburst > 0)
    assert np.all(gal_fburst < 0.1)
    lgyr_peak = gal_burstshape_params[:, 0]
    lgyr_max = gal_burstshape_params[:, 1]
    assert np.all(lgyr_peak > LGYR_PEAK_MIN)
    assert np.all(lgyr_max > lgyr_peak + DLGAGE_MIN)
    assert np.all(lgyr_max < LGAGE_MAX)

    assert gal_frac_bulge_t_obs.shape == (n_gals,)
    assert np.all(gal_frac_bulge_t_obs > 0)
    assert np.all(gal_frac_bulge_t_obs < 1)
    assert gal_fbulge_params.shape == (n_gals, 3)
    assert gal_fknot.shape == (n_gals,)
    assert np.all(gal_fknot > 0)
    assert np.all(gal_fknot < FKNOT_MAX)

    assert gal_obsmags_nodust.shape == (n_gals, n_obs_filters)
    assert gal_restmags_nodust.shape == (n_gals, n_rest_filters)
    assert gal_obsmags_dust.shape == (n_gals, n_obs_filters)
    assert gal_restmags_dust.shape == (n_gals, n_rest_filters)

    assert np.all(gal_obsmags_dust >= gal_obsmags_nodust)
    assert np.any(gal_obsmags_dust > gal_obsmags_nodust)

    assert np.all(gal_restmags_dust >= gal_restmags_nodust)
    assert np.any(gal_restmags_dust > gal_restmags_nodust)
