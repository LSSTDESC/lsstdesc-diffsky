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
from dsps.experimental.diffburst import (
    DLGAGE_MIN,
    LGAGE_MAX,
    LGYR_PEAK_MIN,
    _age_weights_from_params,
)
from dsps.utils import _jax_get_dt_array
from jax import jit as jjit
from jax import random as jran
from jax import vmap

from ...disk_bulge_modeling.disk_knots import FKNOT_MAX
from ...disk_bulge_modeling.mc_disk_bulge import mc_disk_bulge
from ..photometry_lc_interp import (
    _burst_age_weights_from_params_vmap,
    _decompose_sfhpop_into_bulge_disk_knots,
    _linterp_vmap,
    decompose_sfhpop_into_bulge_disk_knots,
    get_diffsky_sed_info,
)

_B = (0, None)
_integrate_sfr_vmap = jjit(vmap(_integrate_sfr, in_axes=_B))


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

    ran_key = jran.PRNGKey(0)
    _res = get_diffsky_sed_info(
        ran_key,
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
        gal_fburst,
        gal_burstshape_params,
        gal_fbulge_params,
        gal_fknot,
        gal_rest_seds,
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

    assert gal_fbulge_params.shape == (n_gals, 3)
    assert gal_fknot.shape == (n_gals,)
    assert np.all(gal_fknot > 0)
    assert np.all(gal_fknot < FKNOT_MAX)

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
    """Enforce physically reasonable disk/bulge/knot decomposition for some random
    galaxy distributions. Run n_tests tests for galaxy populations with
    different distributions of {Fburst, t_obs}

    """
    ran_key = jran.PRNGKey(0)

    n_age = 40
    ssp_lg_age_yr = np.linspace(5, 10.25, n_age)
    ssp_lg_age_gyr = ssp_lg_age_yr - 9.0

    n_t = 100
    t0 = 13.8
    t_table_min = 0.01
    gal_t_table = np.linspace(t_table_min, t0, n_t)

    n_gals = 1_000
    n_tests = 20
    for itest in range(n_tests):
        itest_key, ran_key = jran.split(ran_key, 2)

        # make sure t_obs > t_table_min
        t_obs_min = 0.2
        gal_t_obs = np.random.uniform(t_obs_min, t0 - 0.05, n_gals)
        sfh_peak_max = np.random.uniform(0, 200)
        sfh_peak = np.random.uniform(0, sfh_peak_max, n_gals)
        gal_sfh_u = np.random.uniform(0, 1, n_gals * n_t).reshape((n_gals, n_t))
        gal_sfh = gal_sfh_u * sfh_peak.reshape((n_gals, 1))

        LGFBURST_UPPER_BOUND = -1
        lgfb_max = np.random.uniform(-3, LGFBURST_UPPER_BOUND)
        lgfb_min = np.random.uniform(lgfb_max - 3, lgfb_max)
        gal_fburst = 10 ** np.random.uniform(lgfb_min, lgfb_max, n_gals)

        gal_burstshape_params = np.tile(DEFAULT_BURST_PARAMS, n_gals)
        gal_burstshape_params = gal_burstshape_params.reshape((n_gals, 2))

        gal_burst_age_weights = _burst_age_weights_from_params_vmap(
            ssp_lg_age_yr, gal_burstshape_params
        )

        age_weights_singleburst = _age_weights_from_params(
            ssp_lg_age_yr, DEFAULT_BURST_PARAMS
        )
        age_weights_burstpop = np.tile(age_weights_singleburst, n_gals)
        age_weights_burstpop = age_weights_burstpop.reshape((n_gals, n_age))
        assert np.allclose(gal_burst_age_weights, age_weights_burstpop, rtol=0.001)

        gal_fknot = np.random.uniform(0, FKNOT_MAX, n_gals)
        gal_fbulge_params = mc_disk_bulge(itest_key, gal_t_table, gal_sfh)[0]

        args = (
            gal_fbulge_params,
            gal_fknot,
            gal_t_obs,
            gal_t_table,
            gal_sfh,
            gal_fburst,
            age_weights_burstpop,
            ssp_lg_age_gyr,
        )
        _res = _decompose_sfhpop_into_bulge_disk_knots(*args)
        _res2 = decompose_sfhpop_into_bulge_disk_knots(
            gal_fbulge_params,
            gal_fknot,
            gal_t_obs,
            gal_t_table,
            gal_sfh,
            gal_fburst,
            gal_burstshape_params,
            ssp_lg_age_gyr,
        )
        # Enforce no NaN and that the convenience function agrees with the kernel
        for x, x2 in zip(_res, _res2):
            assert np.all(np.isfinite(x))
            assert np.allclose(x, x2, rtol=1e-4)

        mbulge, mdd, mknot, mburst = _res[:4]
        bulge_age_weights, dd_age_weights, knot_age_weights = _res[4:7]
        bulge_sfh, frac_bulge_t_obs = _res[7:]

        assert mbulge.shape == (n_gals,)
        assert mdd.shape == (n_gals,)
        assert mknot.shape == (n_gals,)
        assert mburst.shape == (n_gals,)
        assert bulge_age_weights.shape == (n_gals, n_age)
        assert dd_age_weights.shape == (n_gals, n_age)
        assert knot_age_weights.shape == (n_gals, n_age)
        assert bulge_sfh.shape == (n_gals, n_t)
        assert frac_bulge_t_obs.shape == (n_gals,)

        # Each galaxy's age weights should be a unit-normalized PDF
        assert np.allclose(np.sum(bulge_age_weights, axis=1), 1.0, rtol=0.01)
        assert np.allclose(np.sum(dd_age_weights, axis=1), 1.0, rtol=0.01)
        assert np.allclose(np.sum(knot_age_weights, axis=1), 1.0, rtol=0.01)

        # The sum of the masses in each component should equal
        # # the total stellar mass formed at gal_t_obs
        mtot = mbulge + mdd + mknot
        lgmtot = np.log10(mtot)

        dt_table = _jax_get_dt_array(gal_t_table)
        gal_smh = np.cumsum(gal_sfh * dt_table, axis=1) * 1e9
        gal_logsmh = np.log10(gal_smh)

        lgt_table = np.log10(gal_t_table)
        lgmstar_t_obs = _linterp_vmap(np.log10(gal_t_obs), lgt_table, gal_logsmh)

        assert np.allclose(lgmstar_t_obs, lgmtot, atol=0.01)

        # Star-forming knots should never have fewer young stars than the diffuse disk
        dd_age_cdf = np.cumsum(dd_age_weights, axis=1)
        knot_age_cdf = np.cumsum(knot_age_weights, axis=1)

        tol = 0.01
        indx_ostar = np.searchsorted(ssp_lg_age_yr, 6.5)
        assert np.all(dd_age_cdf[:, indx_ostar] <= knot_age_cdf[:, indx_ostar] + tol)
        assert np.any(dd_age_cdf[:, indx_ostar] < knot_age_cdf[:, indx_ostar])

        # The bulge should never have more young stars than star-forming knots
        tol = 0.01
        bulge_age_cdf = np.cumsum(bulge_age_weights, axis=1)
        assert np.all(bulge_age_cdf[:, indx_ostar] <= knot_age_cdf[:, indx_ostar] + tol)

        # On average, star-forming knots should be younger than diffuse disks
        # and diffuse disks should should be younger than bulges
        bulge_median_frac_ob_stars = np.median(bulge_age_cdf[:, indx_ostar])
        dd_median_frac_ob_stars = np.median(dd_age_cdf[:, indx_ostar])
        knot_median_frac_ob_stars = np.median(knot_age_cdf[:, indx_ostar])
        assert (
            bulge_median_frac_ob_stars
            < dd_median_frac_ob_stars
            < knot_median_frac_ob_stars
        )

        # Bulge SFH should never exceed total SFH
        assert np.all(bulge_sfh <= gal_sfh)

        # Bulge SFH should fall below total SFH at some point in history of each galaxy
        assert np.all(np.any(bulge_sfh < gal_sfh, axis=1))

        # Bulge fraction should respect 0 < frac_bulge < 1 for every galaxy
        assert np.all(frac_bulge_t_obs > 0)
        assert np.all(frac_bulge_t_obs < 1)
        # Bulge fraction should respect 0 < frac_bulge < 1 for every galaxy
        assert np.all(frac_bulge_t_obs > 0)
        assert np.all(frac_bulge_t_obs < 1)
        assert np.all(frac_bulge_t_obs < 1)
        assert np.all(frac_bulge_t_obs < 1)
