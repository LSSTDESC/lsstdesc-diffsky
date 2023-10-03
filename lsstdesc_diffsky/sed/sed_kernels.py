"""
"""
from diffstar import sfh_singlegal
from dsps.constants import N_T_LGSM_INTEGRATION, T_BIRTH_MIN
from dsps.cosmology.flat_wcdm import _age_at_z_kern, age_at_z0
from dsps.metallicity.mzr import mzr_model
from dsps.sed.metallicity_weights import calc_lgmet_weights_from_lognormal_mdf
from dsps.sed.stellar_age_weights import _calc_age_weights_from_logsm_table
from dsps.utils import _jax_get_dt_array
from jax import jit as jjit
from jax import numpy as jnp

from ..defaults import DEFAULT_COSMO_PARAMS


@jjit
def calc_rest_sed_singlegal(
    z_obs,
    diffmah_params,
    diffstar_ms_params,
    diffstar_q_params,
    met_params,
    ssp_lgmet,
    ssp_lg_age_gyr,
    ssp_flux,
    t_birth_min=T_BIRTH_MIN,
    n_t=N_T_LGSM_INTEGRATION,
    cosmo_params=DEFAULT_COSMO_PARAMS,
):
    Om0, w0, wa, h, fb = cosmo_params
    t0 = age_at_z0(Om0, w0, wa, h)
    t_table = jnp.linspace(2 * t_birth_min, t0, n_t)
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

    mzr_params, lgmet_scatter = met_params[:-1], met_params[-1]
    _galprops_at_t_obs = _get_galprops_at_t_obs_singlegal(
        t_obs, t_table, sfh_table, mzr_params, ssp_lg_age_gyr
    )
    logsm_t_obs, logssfr_t_obs, lgmet_t_obs, smooth_age_weights = _galprops_at_t_obs[:4]
    lgmet_weights = calc_lgmet_weights_from_lognormal_mdf(
        lgmet_t_obs, lgmet_scatter, ssp_lgmet
    )
    return logsm_t_obs, lgmet_t_obs, lgmet_weights, smooth_age_weights


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

    sfr_table_age_weights = _calc_age_weights_from_logsm_table(
        lgt_table, logsmh_table, ssp_lg_age_gyr, t_obs
    )

    return (
        logsm_t_obs,
        logssfr_t_obs,
        lgmet_t_obs,
        sfr_table_age_weights,
    )
