"""
"""
from diffstar import sfh_singlegal
from dsps.constants import N_T_LGSM_INTEGRATION, T_BIRTH_MIN
from dsps.cosmology.flat_wcdm import _age_at_z_kern, age_at_z0
from dsps.dust.att_curves import (
    UV_BUMP_DW,
    UV_BUMP_W0,
    _frac_transmission_from_k_lambda,
    _get_eb_from_delta,
    sbl18_k_lambda,
)
from dsps.experimental.diffburst import (
    _age_weights_from_u_params as _burst_age_weights_from_u_params,
)
from dsps.metallicity.mzr import mzr_model
from dsps.sed.metallicity_weights import calc_lgmet_weights_from_lognormal_mdf
from dsps.sed.stellar_age_weights import _calc_age_weights_from_logsm_table
from dsps.utils import _jax_get_dt_array
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..defaults import DEFAULT_COSMO_PARAMS
from ..dspspop.boris_dust import _get_funo_from_u_params_singlegal
from ..dspspop.burstshapepop import _get_burstshape_galpop_from_params
from ..dspspop.dust_deltapop import _get_dust_delta_galpop_from_u_params
from ..dspspop.lgavpop import _get_lgav_galpop_from_u_params
from ..dspspop.lgfburstpop import _get_lgfburst_galpop_from_u_params

_T = (None, None, 0)
_frac_transmission_from_k_lambda_age_vmap = jjit(
    vmap(_frac_transmission_from_k_lambda, in_axes=_T)
)


@jjit
def calc_rest_sed_singlegal(
    z_obs,
    diffmah_params,
    diffstar_ms_params,
    diffstar_q_params,
    ssp_data,
    diffskypop_params,
    cosmo_params=DEFAULT_COSMO_PARAMS,
):
    """Calculate the restframe SED of an individual diffsky galaxy

    Parameters
    ----------
    z_obs : float
        Redshift of the galaxy

    diffmah_params : ndarray, shape (4, )
        Diffmah params specifying the mass assembly of the dark matter halo
        diffmah_params = (logm0, logtc, early_index, late_index)

    diffstar_ms_params : ndarray, shape (5, )
        Diffstar params for the star-formation effiency and gas consumption timescale
        ms_params = (lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep)

    diffstar_q_params : ndarray, shape (4, )
        Diffstar quenching params, (lg_qt, qlglgdt, lg_drop, lg_rejuv)

    ssp_lgmet : ndarray, shape (n_met, )
        Grid in metallicity Z at which the SSPs are computed, stored as log10(Z)

    ssp_lg_age_gyr : ndarray, shape (n_age, )
        Grid in age τ at which the SSPs are computed, stored as log10(τ/Gyr)

    ssp_wave_ang : ndarray, shape (n_wave, )
        Array of wavelengths in angstroms at which the SSP SEDs are tabulated

    ssp_flux : ndarray, shape (n_met, n_age, n_wave)
        Tabulation of the SSP SEDs at the input wavelength in Lsun/Hz/Msun

    lgfburst_pop_u_params : ndarray, shape (n_pars_lgfburst_pop, )
        Unbounded parameters controlling Fburst, which sets the fractional contribution
        of a recent burst to the smooth SFH of a galaxy. For typical values, see
        dspspop.lgfburstpop.DEFAULT_LGFBURST_U_PARAMS

    burstshapepop_u_params : ndarray, shape (n_pars_burstshape_pop, )
        Unbounded parameters controlling the distribution of stellar ages
        of stars formed in a recent burst. For typical values, see
        dspspop.burstshapepop.DEFAULT_BURSTSHAPE_U_PARAMS

    lgav_pop_u_params : ndarray, shape (n_pars_lgav_pop, )
        Unbounded parameters controlling the distribution of dust parameter Av,
        the normalization of the attenuation curve at λ_V=5500 angstrom.
        For typical values, see
        dspspop.lgavpop.DEFAULT_LGAV_U_PARAMS

    dust_delta_pop_u_params : ndarray, shape (n_pars_dust_delta_pop, )
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

    cosmo_params : optional, ndarray, shape (5, )
        cosmo_params = (Om0, w0, wa, h, fb)
        Defaults set in lsstdesc_diffsky.defaults.DEFAULT_COSMO_PARAMS

    Returns
    -------
    rest_sed : ndarray, shape (n_wave, )
        Restframe SED of the galaxy in units of Lsun/Hz

    rest_sed_nodust : ndarray, shape (n_wave, )
        Restframe SED of the galaxy in units of Lsun/Hz, neglecting dust attenuation

    logsm_t_obs : float
        Total stellar mass formed at z_obs in units of Msun

    lgmet_t_obs : float
        Stellar metallicity at z_obs (dimensionless)

    """
    Om0, w0, wa, h, fb = cosmo_params
    t0 = age_at_z0(Om0, w0, wa, h)
    t_table = jnp.linspace(2 * T_BIRTH_MIN, t0, N_T_LGSM_INTEGRATION)
    lgt0 = jnp.log10(t0)
    t_obs = _age_at_z_kern(z_obs, Om0, w0, wa, h)
    sfh_table = sfh_singlegal(
        t_table,
        diffmah_params,
        diffstar_ms_params,
        diffstar_q_params,
        lgt0=lgt0,
        fb=fb,
    )

    mzr_params = diffskypop_params.lgmet_params[:-1]
    lgmet_scatter = diffskypop_params.lgmet_params[-1]

    _galprops_at_t_obs = _get_galprops_at_t_obs_singlegal(
        t_obs, t_table, sfh_table, mzr_params, ssp_data.ssp_lg_age_gyr
    )
    logsm_t_obs, logssfr_t_obs, lgmet_t_obs, smooth_age_weights = _galprops_at_t_obs[:4]
    mstar_t_obs = 10**logsm_t_obs

    lgmet_weights = calc_lgmet_weights_from_lognormal_mdf(
        lgmet_t_obs, lgmet_scatter, ssp_data.ssp_lgmet
    )

    # Compute burst fraction for every galaxy
    lgfburst = _get_lgfburst_galpop_from_u_params(
        logsm_t_obs, logssfr_t_obs, diffskypop_params.lgfburst_u_params
    )
    fburst = 10**lgfburst

    # Compute P(τ) for each bursting population
    gal_u_lgyr_peak, gal_u_lgyr_max = _get_burstshape_galpop_from_params(
        logsm_t_obs, logssfr_t_obs, diffskypop_params.burstshape_u_params
    )
    burstshape_u_params = jnp.array((gal_u_lgyr_peak, gal_u_lgyr_max)).T
    ssp_lg_age_yr = ssp_data.ssp_lg_age_gyr + 9
    burst_age_weights = _burst_age_weights_from_u_params(
        ssp_lg_age_yr, burstshape_u_params
    )

    # Compute P(τ) for each composite galaxy
    age_weights = fburst * burst_age_weights + (1 - fburst) * smooth_age_weights

    n_met, n_age, n_wave = ssp_data.ssp_flux.shape
    weights = lgmet_weights.reshape((n_met, 1, 1)) * age_weights.reshape((1, n_age, 1))
    rest_sed_nodust = jnp.sum(weights * ssp_data.ssp_flux, axis=(0, 1)) * mstar_t_obs

    lgav = _get_lgav_galpop_from_u_params(
        logsm_t_obs, logssfr_t_obs, diffskypop_params.lgav_dust_u_params
    )
    dust_delta = _get_dust_delta_galpop_from_u_params(
        logsm_t_obs, logssfr_t_obs, diffskypop_params.delta_dust_u_params
    )
    frac_unobscured = _get_funo_from_u_params_singlegal(
        logsm_t_obs,
        lgfburst,
        logssfr_t_obs,
        ssp_data.ssp_lg_age_gyr,
        diffskypop_params.funo_dust_u_params,
    )

    ssp_wave_micron = ssp_data.ssp_wave / 1e4
    dust_Av = 10**lgav
    dust_Eb = _get_eb_from_delta(dust_delta)
    k_lambda = sbl18_k_lambda(
        ssp_wave_micron, UV_BUMP_W0, UV_BUMP_DW, dust_Eb, dust_delta
    )
    frac_dust_trans = _frac_transmission_from_k_lambda_age_vmap(
        k_lambda, dust_Av, frac_unobscured
    )
    frac_dust_trans = frac_dust_trans.reshape((1, n_age, n_wave))
    rest_sed = (
        jnp.sum(weights * ssp_data.ssp_flux * frac_dust_trans, axis=(0, 1))
        * mstar_t_obs
    )

    return (rest_sed, rest_sed_nodust, logsm_t_obs, lgmet_t_obs)


@jjit
def _get_galprops_at_t_obs_singlegal(
    t_obs, t_table, sfr_table, mzr_params, ssp_lg_age_gyr
):
    lgt_obs = jnp.log10(t_obs)
    lgt_table = jnp.log10(t_table)

    dt_table = _jax_get_dt_array(t_table)
    logsmh_table = jnp.log10(jnp.cumsum(sfr_table * dt_table)) + 9.0

    logsfr_table = jnp.log10(sfr_table)

    logsm_t_obs = jnp.interp(lgt_obs, lgt_table, logsmh_table)
    logsfr_t_obs = jnp.interp(lgt_obs, lgt_table, logsfr_table)
    logssfr_t_obs = logsfr_t_obs - logsm_t_obs

    lgmet_t_obs = mzr_model(logsm_t_obs, t_obs, *mzr_params)

    __, sfr_table_age_weights = _calc_age_weights_from_logsm_table(
        lgt_table, logsmh_table, ssp_lg_age_gyr, t_obs
    )

    return (
        logsm_t_obs,
        logssfr_t_obs,
        lgmet_t_obs,
        sfr_table_age_weights,
    )


_G = (0, *[0] * 3, *[None] * 3)
_calc_rest_sed_vmap = jjit(vmap(calc_rest_sed_singlegal, in_axes=_G))


@jjit
def calc_rest_sed_galpop(
    z_obs,
    diffmah_params,
    diffstar_ms_params,
    diffstar_q_params,
    ssp_data,
    diffskypop_params,
    cosmo_params=DEFAULT_COSMO_PARAMS,
):
    """Calculate the restframe SED of a population of diffsky galaxies

    Parameters
    ----------
    z_obs : ndarray, shape (n_gals, )
        Redshift of the galaxies

    diffmah_params : ndarray, shape (n_gals, 4)
        Diffmah params specifying the mass assembly of the dark matter halo
        diffmah_params = (logm0, logtc, early_index, late_index)

    diffstar_ms_params : ndarray, shape (n_gals, 5)
        Diffstar params for the star-formation effiency and gas consumption timescale
        ms_params = (lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep)

    diffstar_q_params : ndarray, shape (n_gals, 4)
        Diffstar quenching params, (lg_qt, qlglgdt, lg_drop, lg_rejuv)

    ssp_data : namedtuple, shape (3, )
        NamedTuple with the following three entries:

        ssp_lgmet : ndarray, shape (n_met, )
            Grid in metallicity Z at which the SSPs are computed, stored as log10(Z)

        ssp_lg_age_gyr : ndarray, shape (n_age, )
            Grid in age τ at which the SSPs are computed, stored as log10(τ/Gyr)

        ssp_wave_ang : ndarray, shape (n_wave, )
            Array of wavelengths in angstroms at which the SSP SEDs are tabulated

        ssp_flux : ndarray, shape (n_age, n_wave)
            Tabulation of the SSP SEDs at the input wavelength in Lsun/Hz/Msun

    lgfburst_pop_u_params : ndarray, shape (n_pars_lgfburst_pop, )
        Unbounded parameters controlling Fburst, which sets the fractional contribution
        of a recent burst to the smooth SFH of a galaxy. For typical values, see
        dspspop.lgfburstpop.DEFAULT_LGFBURST_U_PARAMS

    burstshapepop_u_params : ndarray, shape (n_pars_burstshape_pop, )
        Unbounded parameters controlling the distribution of stellar ages
        of stars formed in a recent burst. For typical values, see
        dspspop.burstshapepop.DEFAULT_BURSTSHAPE_U_PARAMS

    lgav_pop_u_params : ndarray, shape (n_pars_lgav_pop, )
        Unbounded parameters controlling the distribution of dust parameter Av,
        the normalization of the attenuation curve at λ_V=5500 angstrom.
        For typical values, see
        dspspop.lgavpop.DEFAULT_LGAV_U_PARAMS

    dust_delta_pop_u_params : ndarray, shape (n_pars_dust_delta_pop, )
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

    cosmo_params : optional, ndarray, shape (5, )
        cosmo_params = (Om0, w0, wa, h, fb)
        Defaults set in lsstdesc_diffsky.defaults.DEFAULT_COSMO_PARAMS

    Returns
    -------
    rest_sed : ndarray, shape (n_gals, n_wave)
        Restframe SED of the galaxy in units of Lsun/Hz

    rest_sed_nodust : ndarray, shape (n_gals, n_wave)
        Restframe SED of the galaxy in units of Lsun/Hz, neglecting dust attenuation

    logsm_t_obs : ndarray, shape (n_gals, )
        Total stellar mass formed at z_obs in units of Msun

    lgmet_t_obs : ndarray, shape (n_gals, )
        Stellar metallicity at z_obs (dimensionless)

    """
    return _calc_rest_sed_vmap(
        z_obs,
        diffmah_params,
        diffstar_ms_params,
        diffstar_q_params,
        ssp_data,
        diffskypop_params,
        cosmo_params,
    )
