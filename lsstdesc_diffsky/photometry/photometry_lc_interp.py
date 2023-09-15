"""
"""
from diffsky.experimental.dspspop.burstshapepop import (
    _get_burstshape_galpop_from_params,
)
from diffsky.experimental.dspspop.dustpop import (
    _frac_dust_transmission_lightcone_kernel,
    _frac_dust_transmission_singlez_kernel,
)
from diffsky.experimental.dspspop.lgfburstpop import _get_lgfburst_galpop_from_u_params
from diffsky.experimental.photometry_interpolation import interpolate_ssp_photmag_table
from diffstar.defaults import SFR_MIN
from dsps.cosmology.flat_wcdm import _age_at_z_vmap
from dsps.experimental.diffburst import _age_weights_from_u_params
from dsps.metallicity.mzr import DEFAULT_MZR_PDICT, mzr_model
from dsps.sed import calc_ssp_weights_sfh_table_lognormal_mdf
from dsps.sed.stellar_age_weights import (
    _calc_logsm_table_from_sfh_table,
    calc_age_weights_from_sfh_table,
)
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..disk_bulge_modeling.disk_knots import FKNOT_MAX, _disk_knot_vmap
from ..disk_bulge_modeling.mc_disk_bulge import mc_disk_bulge

_D = (None, 0, None, 0)
calc_age_weights_from_sfh_table_vmap = jjit(
    vmap(calc_age_weights_from_sfh_table, in_axes=_D)
)

_linterp_vmap = jjit(vmap(jnp.interp, in_axes=(0, None, 0)))

_g = (None, 0, 0, None, None, None, 0)
calc_ssp_weights_sfh_table_lognormal_mdf_vmap = jjit(
    vmap(calc_ssp_weights_sfh_table_lognormal_mdf, in_axes=_g)
)

_b = (None, 0, None)
_calc_logsm_table_from_sfh_table_vmap = jjit(
    vmap(_calc_logsm_table_from_sfh_table, in_axes=_b)
)

_A = (None, 0)
_age_weights_from_u_params_vmap = jjit(vmap(_age_weights_from_u_params, in_axes=_A))

DEFAULT_MZR_PARAMS = jnp.array(list(DEFAULT_MZR_PDICT.values()))


def get_diffsky_sed_info(
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
    lgfburst_pop_u_params,
    burstshapepop_u_params,
    lgav_u_params,
    dust_delta_u_params,
    fracuno_pop_u_params,
    met_params=DEFAULT_MZR_PARAMS,
    lgmet_scatter=0.2,
):
    _check_ssp_info_shapes(ssp_z_table, gal_z_obs)

    ssp_obsmag_table_pergal = _get_ssp_obsmag_table_pergal(
        gal_z_obs, ssp_z_table, ssp_obsmag_table, ssp_restmag_table, gal_sfr_table
    )
    ssp_obsflux_table_pergal = 10 ** (-0.4 * ssp_obsmag_table_pergal)
    ssp_restflux_table = 10 ** (-0.4 * ssp_restmag_table)

    n_gals, n_met, n_age, n_obs_filters = ssp_obsmag_table_pergal.shape
    n_rest_filters = ssp_restmag_table.shape[-1]

    _res = _get_galprops_at_t_obs(
        gal_z_obs, gal_t_table, gal_sfr_table, met_params, cosmo_params
    )
    gal_t_obs, gal_logsm_t_obs, gal_logssfr_t_obs, gal_lgmet_t_obs = _res

    args = (
        gal_t_table,
        gal_sfr_table,
        gal_lgmet_t_obs,
        lgmet_scatter,
        ssp_lgmet,
        ssp_lg_age_gyr,
        gal_t_obs,
    )
    _res = calc_ssp_weights_sfh_table_lognormal_mdf_vmap(*args)
    lgmet_weights, smooth_age_weights = _res[1:]

    gal_lgf_burst = _get_lgfburst_galpop_from_u_params(
        gal_logsm_t_obs, gal_logssfr_t_obs, lgfburst_pop_u_params
    )
    gal_fburst = 10**gal_lgf_burst

    burstshape_u_params = _get_burstshape_galpop_from_params(
        gal_logsm_t_obs, gal_logssfr_t_obs, burstshapepop_u_params
    )
    burstshape_u_params = jnp.array(burstshape_u_params).T
    ssp_lg_age_yr = ssp_lg_age_gyr + 9
    burst_weights = _age_weights_from_u_params_vmap(ssp_lg_age_yr, burstshape_u_params)

    _fb = gal_fburst.reshape((n_gals, 1))
    bursty_age_weights = _fb * burst_weights + (1 - _fb) * smooth_age_weights

    _w_age = bursty_age_weights.reshape((n_gals, 1, n_age))
    _w_met = lgmet_weights.reshape((n_gals, n_met, 1))
    _w = _w_age * _w_met
    _norm = jnp.sum(_w, axis=(1, 2))
    gal_weights = _w / _norm.reshape((n_gals, 1, 1))  # (n_gals, n_met, n_age)
    gal_weights = gal_weights.reshape((n_gals, n_met, n_age, 1))

    gal_mstar_obs = (10**gal_logsm_t_obs).reshape((n_gals, 1))

    prod_rest_seds = gal_weights * ssp_rest_seds
    gal_rest_seds = jnp.sum(prod_rest_seds, axis=(1, 2)) * gal_mstar_obs

    prod_obs_nodust = gal_weights * ssp_obsflux_table_pergal
    gal_obsflux_nodust = jnp.sum(prod_obs_nodust, axis=(1, 2)) * gal_mstar_obs
    gal_obsmags_nodust = -2.5 * jnp.log10(gal_obsflux_nodust)

    prod_rest = gal_weights * ssp_restflux_table
    gal_restflux_nodust = jnp.sum(prod_rest, axis=(1, 2)) * gal_mstar_obs
    gal_restmags_nodust = -2.5 * jnp.log10(gal_restflux_nodust)

    dummy_dust_key = 0
    _res = _frac_dust_transmission_lightcone_kernel(
        dummy_dust_key,
        gal_z_obs,
        gal_logsm_t_obs,
        gal_logssfr_t_obs,
        gal_lgf_burst,
        ssp_lg_age_gyr,
        obs_filter_waves,
        obs_filter_trans,
        lgav_u_params,
        dust_delta_u_params,
        fracuno_pop_u_params,
    )
    gal_frac_trans_obs = _res[0]  # (n_gals, n_age, n_filters)
    gal_att_curve_params, gal_frac_unobs = _res[1:]

    ft_obs = gal_frac_trans_obs.reshape((n_gals, 1, n_age, n_obs_filters))
    prod_obs_dust = gal_weights * ssp_obsflux_table_pergal * ft_obs
    gal_obsflux_dust = jnp.sum(prod_obs_dust, axis=(1, 2)) * gal_mstar_obs
    gal_obsmags_dust = -2.5 * jnp.log10(gal_obsflux_dust)

    _res = _frac_dust_transmission_singlez_kernel(
        dummy_dust_key,
        0.0,
        gal_logsm_t_obs,
        gal_logssfr_t_obs,
        gal_lgf_burst,
        ssp_lg_age_gyr,
        rest_filter_waves,
        rest_filter_trans,
        lgav_u_params,
        dust_delta_u_params,
        fracuno_pop_u_params,
    )
    gal_frac_trans_rest = _res[0]  # (n_gals, n_age, n_filters)

    ft_rest = gal_frac_trans_rest.reshape((n_gals, 1, n_age, n_rest_filters))
    prod_rest_dust = gal_weights * ssp_restflux_table * ft_rest
    gal_restflux_dust = jnp.sum(prod_rest_dust, axis=(1, 2)) * gal_mstar_obs
    gal_restmags_dust = -2.5 * jnp.log10(gal_restflux_dust)

    return (
        gal_weights,
        gal_frac_trans_obs,
        gal_frac_trans_rest,
        gal_att_curve_params,
        gal_frac_unobs,
        gal_rest_seds,
        gal_obsmags_nodust,
        gal_restmags_nodust,
        gal_obsmags_dust,
        gal_restmags_dust,
    )


def decompose_sfh_into_bulge_disk_knots(
    ran_key, gal_t_obs, gal_t_table, gal_sfh, fburst, age_weights_burst, ssp_lg_age_gyr
):
    bulge_key, knot_key = jran.split(ran_key, 2)

    gal_smh, eff_bulge, bulge_sfh, bulge_smh, bth = mc_disk_bulge(
        bulge_key, gal_t_table, gal_sfh
    )
    bulge_sfh = jnp.where(bulge_sfh < SFR_MIN, SFR_MIN, bulge_sfh)
    frac_bulge_t_obs = _linterp_vmap(gal_t_obs, gal_t_table, bth)

    bulge_age_weights = calc_age_weights_from_sfh_table_vmap(
        gal_t_table, bulge_sfh, ssp_lg_age_gyr, gal_t_obs
    )

    disk_sfh = gal_sfh - bulge_sfh
    disk_sfh = jnp.where(disk_sfh < SFR_MIN, SFR_MIN, disk_sfh)

    n_gals = gal_sfh.shape[0]
    fknot = jran.uniform(knot_key, minval=0, maxval=FKNOT_MAX, shape=(n_gals,))

    args = (
        gal_t_table,
        gal_t_obs,
        gal_sfh,
        disk_sfh,
        fburst,
        fknot,
        age_weights_burst,
        ssp_lg_age_gyr,
    )
    _res = _disk_knot_vmap(*args)
    mstar_tot, mburst, mdd, mknot, dd_age_weights, knot_age_weights = _res

    mbulge = frac_bulge_t_obs * mstar_tot
    masses = mbulge, mdd, mknot, mburst
    age_weights = bulge_age_weights, dd_age_weights, knot_age_weights
    ret = (*masses, *age_weights)
    return ret


def _get_galprops_at_t_obs(
    gal_z_obs, gal_t_table, gal_sfr_table, met_params, cosmo_params
):
    gal_t_obs = _age_at_z_vmap(gal_z_obs, *cosmo_params)
    lgt_obs = jnp.log10(gal_t_obs)
    lgt_table = jnp.log10(gal_t_table)

    gal_sfr_table = jnp.where(gal_sfr_table < SFR_MIN, SFR_MIN, gal_sfr_table)
    gal_logsm_table = _calc_logsm_table_from_sfh_table_vmap(
        gal_t_table, gal_sfr_table, SFR_MIN
    )
    gal_logsfr_table = jnp.log10(gal_sfr_table)

    gal_logsm_t_obs = _linterp_vmap(lgt_obs, lgt_table, gal_logsm_table)
    gal_logsfr_t_obs = _linterp_vmap(lgt_obs, lgt_table, gal_logsfr_table)
    gal_logssfr_t_obs = gal_logsfr_t_obs - gal_logsm_t_obs

    gal_lgmet_t_obs = mzr_model(gal_logsm_t_obs, gal_t_obs, *met_params[:-1])

    return gal_t_obs, gal_logsm_t_obs, gal_logssfr_t_obs, gal_lgmet_t_obs


def _check_ssp_info_shapes(ssp_z_table, gal_z_obs):
    msg = "ssp_z_table must be monotonically increasing"
    assert jnp.all(jnp.diff(ssp_z_table) > 0), msg

    msg = "Must have ssp_z_table.min() < gal_z_obs.min()"
    assert jnp.all(ssp_z_table.min() < gal_z_obs.min()), msg

    msg = "Must have ssp_z_table.max() > gal_z_obs.max()"
    assert jnp.all(ssp_z_table.max() > gal_z_obs.max()), msg


def _get_ssp_obsmag_table_pergal(
    gal_z_obs, ssp_z_table, ssp_obsmag_table, ssp_restmag_table, gal_sfr_table
):
    ssp_obsmag_table_pergal = interpolate_ssp_photmag_table(
        gal_z_obs, ssp_z_table, ssp_obsmag_table
    )
    n_gals, n_met, n_age, n_obs_filters = ssp_obsmag_table_pergal.shape

    msg = "gal_sfr_table.shape[0]={0} must equal gal_z_obs.shape[0]={1}"
    _n_gals = gal_sfr_table.shape[0]
    assert n_gals == gal_sfr_table.shape[0], msg.format(n_gals, _n_gals)

    msg = "ssp_lgmet.shape[0]={0} must equal ssp_obsmag_table_pergal.shape[1]={1}"
    _n_met = ssp_obsmag_table_pergal.shape[1]
    assert n_met == _n_met, msg.format(n_met, _n_met)

    msg = "ssp_lg_age_gyr.shape[0]={0} must equal ssp_obsmag_table_pergal.shape[2]={1}"
    _n_age = ssp_obsmag_table_pergal.shape[2]
    assert n_age == _n_age, msg.format(n_age, _n_age)

    n_met2, n_age2, n_rest_filters = ssp_restmag_table.shape
    msg = "ssp_obsmag_table.shape[1]={0} must equal ssp_restmag_table.shape[0]={1}"
    assert n_met == n_met2, msg.format(n_met, n_met2)

    msg = "ssp_obsmag_table.shape[2]={0} must equal ssp_restmag_table.shape[1]={1}"
    assert n_age == n_age2, msg.format(n_age, n_age2)

    return ssp_obsmag_table_pergal
