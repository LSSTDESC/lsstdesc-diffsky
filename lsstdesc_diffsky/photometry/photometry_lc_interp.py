"""Module implements two functions:

1. get_diffsky_sed_info calculates composite the composite SED and photometry of
a population of diffsky galaxies

2. decompose_sfh_into_bulge_disk_knots decomposes the composite SED into 3 components:
bulge, diffuse disk, star-forming knots

"""
import typing

from diffstar import sfh_galpop
from diffstar.defaults import SFR_MIN
from dsps.cosmology.flat_wcdm import _age_at_z_vmap, age_at_z0
from dsps.experimental.diffburst import (
    _age_weights_from_u_params as _burst_age_weights_from_u_params,
)
from dsps.experimental.diffburst import (
    _get_params_from_u_params as _get_burst_params_from_u_params,
)
from dsps.metallicity.mzr import mzr_model
from dsps.sed import calc_ssp_weights_sfh_table_lognormal_mdf
from dsps.sed.stellar_age_weights import _calc_logsm_table_from_sfh_table
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran
from jax import vmap

from ..disk_bulge_modeling.disk_bulge_kernels import calc_tform_pop
from ..disk_bulge_modeling.disk_knots import FKNOT_MAX
from ..disk_bulge_modeling.mc_disk_bulge import _bulge_sfh_vmap, generate_fbulge_params
from ..dspspop.burstshapepop import _get_burstshape_galpop_from_params
from ..dspspop.dustpop import (
    _frac_dust_transmission_lightcone_kernel,
    _frac_dust_transmission_singlez_kernel,
)
from ..dspspop.lgfburstpop import _get_lgfburst_galpop_from_u_params
from ..photometry_interpolation import interpolate_ssp_photmag_table

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
_burst_age_weights_from_u_params_vmap = jjit(
    vmap(_burst_age_weights_from_u_params, in_axes=_A)
)

_get_burst_params_from_u_params_vmap = jjit(vmap(_get_burst_params_from_u_params))


class DiffskySEDinfo(typing.NamedTuple):
    gal_ssp_weights: jnp.ndarray
    gal_frac_trans_obs: jnp.ndarray
    gal_frac_trans_rest: jnp.ndarray
    gal_att_curve_params: jnp.ndarray
    gal_frac_unobs: jnp.ndarray
    gal_fburst: jnp.ndarray
    gal_burstshape_params: jnp.ndarray
    gal_frac_bulge_t_obs: jnp.ndarray
    gal_fbulge_params: jnp.ndarray
    gal_fknot: jnp.ndarray
    gal_obsmags_nodust: jnp.ndarray
    gal_restmags_nodust: jnp.ndarray
    gal_obsmags_dust: jnp.ndarray
    gal_restmags_dust: jnp.ndarray


def get_diffsky_sed_info(
    ran_key,
    gal_z_obs,
    mah_params,
    ms_params,
    q_params,
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
):
    """Compute SED and photometry for population of Diffsky galaxies

    Parameters
    ----------
    ran_key : jax.random.PRNGKey
        Random number seed used to assign values for disk/bulge/knot decomposition

    gal_z_obs : ndarray, shape (n_gals, )
        Redshift of each galaxy

    mah_params : ndarray, shape (n_gals, 4)
        Diffmah params specifying the mass assembly of the dark matter halo
        diffmah_params = (logm0, logtc, early_index, late_index)

    ms_params : ndarray, shape (n_gals, 5)
        Diffstar params for the star-formation effiency and gas consumption timescale
        ms_params = (lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep)

    q_params : ndarray, shape (n_gals, 4)
        Diffstar quenching params, (lg_qt, qlglgdt, lg_drop, lg_rejuv)

    ssp_z_table : ndarray, shape (n_z_table, )
        Table storing a grid in redshift at which SSP photometry have been precomputed

    ssp_restmag_table : ndarray, shape (n_met, n_age, n_rest_filters)
        Restframe AB magnitude of SSPs integrated across input transmission curves
        for n_rest_filters filters

    ssp_obsmag_table : ndarray, shape (n_z_table, n_met, n_age, n_obs_filters)
        Apparent AB magnitude of SSPs observed on a redshift grid with n_z_table points
        for n_obs_filters filters

    ssp_lgmet : ndarray, shape (n_met, )
        Grid in metallicity Z at which the SSPs are computed, stored as log10(Z)

    ssp_lg_age_gyr : ndarray, shape (n_age, )
        Grid in age τ at which the SSPs are computed, stored as log10(τ/Gyr)

    gal_t_table : ndarray, shape (n_t, )
        Grid in cosmic time t in Gyr at which SFH of the galaxy population is tabulated
        gal_t_table should increase monotonically and it should span the
        full range of gal_t_obs, including some padding of a few million years

    rest_filter_waves : ndarray, shape (n_rest_filters, n_trans_wave)
        Grid in λ in angstroms at which n_rest_filters filter transmission
        curves are defined.

        Note that each observer- and rest-frame filter transmission curve must be
        specified by a λ-grid with the same number of points.

    rest_filter_trans : ndarray, shape (n_rest_filters, n_trans_wave)
        Transmission curves of n_rest_filters for photometry restframe absolute mags

    obs_filter_waves : ndarray, shape (n_obs_filters, n_trans_wave)
        Grid in λ in angstroms at which n_obs_filters filter transmission
        curves are defined

        Note that each observer- and rest-frame filter transmission curve must be
        specified by a λ-grid with the same number of points.

    obs_filter_trans : ndarray, shape (n_obs_filters, n_trans_wave)
        Transmission curves of n_obs_filters for photometry apparent mags

    lgfburst_pop_u_params : ndarray, shape (n_pars_lgfburst_pop, )
        Unbounded parameters controlling Fburst, which sets the fractional contribution
        of a recent burst to the smooth SFH of a galaxy. For typical values, see
        dspspop.lgfburstpop.DEFAULT_LGFBURST_U_PARAMS

    burstshapepop_u_params : ndarray, shape (n_pars_burstshape_pop, )
        Unbounded parameters controlling the distribution of stellar ages
        of stars formed in a recent burst. For typical values, see
        dspspop.burstshapepop.DEFAULT_BURSTSHAPE_U_PARAMS

    lgav_u_params : ndarray, shape (n_pars_lgav_pop, )
        Unbounded parameters controlling the distribution of dust parameter Av,
        the normalization of the attenuation curve at λ_V=5500 angstrom.
        For typical values, see
        dspspop.lgavpop.DEFAULT_LGAV_U_PARAMS

    dust_delta_u_params : ndarray, shape (n_pars_dust_delta_pop, )
        Unbounded parameters controlling the distribution of dust parameter δ,
        which modifies the power-law slope of the attenuation curve. For typical values,
        see dspspop.dust_deltapop.DEFAULT_DUST_DELTA_U_PARAMS

    fracuno_pop_u_params : ndarray, shape (n_pars_fracuno_pop, )
        Unbounded parameters controlling the fraction of sightlines unobscured by dust.
        For typical values, see dspspop.boris_dust.DEFAULT_U_PARAMS

    met_params : ndarray, shape (n_pars_met_pop, ), optional
        Parameters controlling the mass-metallicity scaling relation.
        For typical values, see dsps.metallicity.mzr.DEFAULT_MZR_PDICT
        mzr_params = met_params[:-1]
        lgmet_scatter = met_params[-1]

    cosmo_params : ndarray, shape (n_cosmo, )
        Defined in lsstdesc_diffsky.defaults, cosmo_params = (Om0, w0, wa, h, fb)

    Returns
    ----------
    gal_weights : ndarray, shape (n_gals, n_met, n_age)
        Probability distribution P(Z, τ_age) for each galaxy

    gal_frac_trans_obs : ndarray, shape (n_gals, n_age, n_obs_filters)
        For each galaxy, array stores the fraction of light transmitted through dust
        as a function of τ_age and filter bandpass used in observer-frame magnitudes

    gal_frac_trans_rest : ndarray, shape (n_gals, n_age, n_rest_filters)
        For each galaxy, array stores the fraction of light transmitted through dust
        as a function of τ_age and filter bandpass used in rest-frame magnitudes

    gal_att_curve_params : ndarray, shape (n_gals, 3)
        Dust attenuation curve parameters (dust_eb, dust_delta, dust_av) for each galaxy

    gal_frac_unobs : ndarray, shape (n_gals, n_age)
        Unobscured fraction for every galaxy at every τ_age

    gal_fburst : ndarray, shape (n_gals, )
        Fraction of the galaxy mass in the bursting population at gal_t_obs

    gal_burstshape_params : ndarray, shape (n_gals, 2)
        Parameters controlling P(τ) for burst population in each galaxy
        lgyr_peak = gal_burstshape_params[:, 0]
        lgyr_max = gal_burstshape_params[:, 1]

    gal_frac_bulge_t_obs : ndarray, shape (n_gals, )
        Bulge/total mass ratio at gal_t_obs for every galaxy

    gal_fbulge_params : ndarray, shape (n_gals, 3)
        Bulge parameters (fbulge_tcrit, fbulge_early, fbulge_late) for each galaxy

    gal_fknot : ndarray, shape (n_gals, )
        Fraction of the disk mass in bursty star-forming knots for each galaxy

    gal_obsmags_nodust : ndarray, shape (n_gals, n_obs_filters)
        Apparent AB magnitude of each galaxy through each filter,
        neglecting dust attenuation

    gal_restmags_nodust : ndarray, shape (n_gals, n_rest_filters)
        Rest-frame AB magnitude of each galaxy through each filter,
        neglecting dust attenuation

    gal_obsmags_dust : ndarray, shape (n_gals, n_obs_filters)
        Apparent AB magnitude of each galaxy through each filter,
        accounting for dust attenuation

    gal_restmags_dust : ndarray, shape (n_gals, n_rest_filters)
        Rest-frame AB magnitude of each galaxy through each filter,
        accounting for dust attenuation

    """
    # Bounds check input arguments and extract array shapes and sizes
    _check_ssp_info_shapes(ssp_z_table, gal_z_obs)

    fb = cosmo_params[-1]
    t0 = age_at_z0(*cosmo_params[:-1])
    lgt0 = jnp.log10(t0)
    gal_sfr_table = sfh_galpop(
        gal_t_table, mah_params, ms_params, q_params, lgt0=lgt0, fb=fb
    )

    ssp_obsmag_table_pergal = _get_ssp_obsmag_table_pergal(
        gal_z_obs, ssp_z_table, ssp_obsmag_table, ssp_restmag_table, gal_sfr_table
    )
    n_gals, n_met, n_age, n_obs_filters = ssp_obsmag_table_pergal.shape
    n_rest_filters = ssp_restmag_table.shape[-1]

    mzr_params = diffskypop_params.lgmet_params[:-1]
    lgmet_scatter = diffskypop_params.lgmet_params[-1]

    # Compute various galaxy properties at z_obs
    _galprops = _get_galprops_at_t_obs(
        gal_z_obs, gal_t_table, gal_sfr_table, mzr_params, cosmo_params
    )
    gal_t_obs, gal_logsm_t_obs, gal_logssfr_t_obs, gal_lgmet_t_obs = _galprops[:4]
    gal_t10, gal_t90, gal_logsm0 = _galprops[4:]

    # Monte Carlo generate morphology parameters
    fbulge_key, knot_key = jran.split(ran_key, 2)
    gal_fbulge_params = generate_fbulge_params(fbulge_key, gal_t10, gal_t90, gal_logsm0)
    gal_fknot = jran.uniform(knot_key, minval=0, maxval=FKNOT_MAX, shape=(n_gals,))

    # Compute P(Z) and P(τ) for every galaxy
    args = (
        gal_t_table,
        gal_sfr_table,
        gal_lgmet_t_obs,
        lgmet_scatter,
        ssp_data.ssp_lgmet,
        ssp_data.ssp_lg_age_gyr,
        gal_t_obs,
    )
    _weights = calc_ssp_weights_sfh_table_lognormal_mdf_vmap(*args)
    lgmet_weights, smooth_age_weights = _weights[1:]

    # Compute burst fraction for every galaxy
    gal_lgf_burst = _get_lgfburst_galpop_from_u_params(
        gal_logsm_t_obs, gal_logssfr_t_obs, diffskypop_params.lgfburst_u_params
    )
    gal_fburst = 10**gal_lgf_burst

    # Compute P(τ) for each bursting population
    gal_u_lgyr_peak, gal_u_lgyr_max = _get_burstshape_galpop_from_params(
        gal_logsm_t_obs, gal_logssfr_t_obs, diffskypop_params.burstshape_u_params
    )
    burstshape_u_params = jnp.array((gal_u_lgyr_peak, gal_u_lgyr_max)).T
    ssp_lg_age_yr = ssp_data.ssp_lg_age_gyr + 9
    burst_age_weights = _burst_age_weights_from_u_params_vmap(
        ssp_lg_age_yr, burstshape_u_params
    )

    gal_lgyr_peak, gal_lgyr_max = _get_burst_params_from_u_params_vmap(
        burstshape_u_params
    )
    gal_burstshape_params = jnp.array((gal_lgyr_peak, gal_lgyr_max)).T

    # Compute P(τ) for each composite galaxy
    _fb = gal_fburst.reshape((n_gals, 1))
    gal_age_weights = _fb * burst_age_weights + (1 - _fb) * smooth_age_weights

    # Compute P(τ, Z) for each composite galaxy
    _w_age = gal_age_weights.reshape((n_gals, 1, n_age))
    _w_met = lgmet_weights.reshape((n_gals, n_met, 1))
    _w = _w_age * _w_met
    _norm = jnp.sum(_w, axis=(1, 2))
    gal_weights = _w / _norm.reshape((n_gals, 1, 1))  # (n_gals, n_met, n_age)
    gal_weights = gal_weights.reshape((n_gals, n_met, n_age, 1))

    # Compute observed stellar mass for each composite galaxy
    gal_mstar_obs = (10**gal_logsm_t_obs).reshape((n_gals, 1))

    # Compute apparent magnitude in each band for each composite galaxy neglecting dust
    ssp_obsflux_table_pergal = 10 ** (-0.4 * ssp_obsmag_table_pergal)
    prod_obs_nodust = gal_weights * ssp_obsflux_table_pergal
    gal_obsflux_nodust = jnp.sum(prod_obs_nodust, axis=(1, 2)) * gal_mstar_obs
    gal_obsmags_nodust = -2.5 * jnp.log10(gal_obsflux_nodust)

    # Compute restframe magnitude in each band for each composite galaxy neglecting dust
    ssp_restflux_table = 10 ** (-0.4 * ssp_restmag_table)
    prod_rest = gal_weights * ssp_restflux_table
    gal_restflux_nodust = jnp.sum(prod_rest, axis=(1, 2)) * gal_mstar_obs
    gal_restmags_nodust = -2.5 * jnp.log10(gal_restflux_nodust)

    # Compute attenuation through each observed filter bandpass
    dummy_dust_key = 0
    _dust_results_obs = _frac_dust_transmission_lightcone_kernel(
        dummy_dust_key,
        gal_z_obs,
        gal_logsm_t_obs,
        gal_logssfr_t_obs,
        gal_lgf_burst,
        ssp_data.ssp_lg_age_gyr,
        obs_filter_waves,
        obs_filter_trans,
        diffskypop_params.lgav_dust_u_params,
        diffskypop_params.delta_dust_u_params,
        diffskypop_params.funo_dust_u_params,
    )
    gal_frac_trans_obs = _dust_results_obs[0]  # (n_gals, n_age, n_filters)
    gal_att_curve_params, gal_frac_unobs = _dust_results_obs[1:]

    # Apply dust attenuation to the apparent magnitude in each band for each galaxy
    ft_obs = gal_frac_trans_obs.reshape((n_gals, 1, n_age, n_obs_filters))
    prod_obs_dust = gal_weights * ssp_obsflux_table_pergal * ft_obs
    gal_obsflux_dust = jnp.sum(prod_obs_dust, axis=(1, 2)) * gal_mstar_obs
    gal_obsmags_dust = -2.5 * jnp.log10(gal_obsflux_dust)

    # Compute attenuation through each restframe filter bandpass
    _dust_results_rest = _frac_dust_transmission_singlez_kernel(
        dummy_dust_key,
        0.0,
        gal_logsm_t_obs,
        gal_logssfr_t_obs,
        gal_lgf_burst,
        ssp_data.ssp_lg_age_gyr,
        rest_filter_waves,
        rest_filter_trans,
        diffskypop_params.lgav_dust_u_params,
        diffskypop_params.delta_dust_u_params,
        diffskypop_params.funo_dust_u_params,
    )
    gal_frac_trans_rest = _dust_results_rest[0]  # (n_gals, n_age, n_filters)

    # Apply dust attenuation to the restframe magnitude in each band for each galaxy
    ft_rest = gal_frac_trans_rest.reshape((n_gals, 1, n_age, n_rest_filters))
    prod_rest_dust = gal_weights * ssp_restflux_table * ft_rest
    gal_restflux_dust = jnp.sum(prod_rest_dust, axis=(1, 2)) * gal_mstar_obs
    gal_restmags_dust = -2.5 * jnp.log10(gal_restflux_dust)

    _res = _bulge_sfh_vmap(gal_t_table, gal_sfr_table, gal_fbulge_params)
    smh, eff_bulge, bulge_sfh, smh_bulge, bulge_to_total_history = _res

    bulge_sfh = jnp.where(bulge_sfh < SFR_MIN, SFR_MIN, bulge_sfh)
    gal_frac_bulge_t_obs = _linterp_vmap(gal_t_obs, gal_t_table, bulge_to_total_history)

    gal_ssp_weights = gal_weights.reshape((n_gals, n_met, n_age))
    sed_info = DiffskySEDinfo(
        gal_ssp_weights,
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
    )
    return sed_info


def _get_galprops_at_t_obs(
    gal_z_obs, gal_t_table, gal_sfr_table, mzr_params, cosmo_params
):
    Om0, w0, wa, h, fb = cosmo_params
    gal_t_obs = _age_at_z_vmap(gal_z_obs, Om0, w0, wa, h)
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

    gal_smh_table = 10**gal_logsm_table
    gal_t10 = calc_tform_pop(gal_t_table, gal_smh_table, 0.1)
    gal_t90 = calc_tform_pop(gal_t_table, gal_smh_table, 0.9)
    gal_logsm0 = gal_smh_table[:, -1]

    gal_lgmet_t_obs = mzr_model(gal_logsm_t_obs, gal_t_obs, *mzr_params)

    return (
        gal_t_obs,
        gal_logsm_t_obs,
        gal_logssfr_t_obs,
        gal_lgmet_t_obs,
        gal_t10,
        gal_t90,
        gal_logsm0,
    )


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
