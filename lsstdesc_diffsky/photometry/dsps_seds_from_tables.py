"""
"""
import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

try:
    from dsps.sed.stellar_age_weights import _calc_age_weights_from_logsm_table
    from dsps.utils import (
        _get_bin_edges,
        _get_triweights_singlepoint,
        _jax_get_dt_array,
    )
    from dsps.constants import LGMET_LO, LGMET_HI

    HAS_DSPS = True
except ImportError:
    HAS_DSPS = False


@jjit
def _calc_age_met_weights_from_sfh_table(
    t_obs,
    ssp_lgmet,
    ssp_lg_age,
    lgt_table,
    logsm_table,
    lgmet,
    lgmet_scatter,
):
    age_weights = _calc_age_weights_from_logsm_table(
        lgt_table, logsm_table, ssp_lg_age, t_obs
    )[1]
    lgmet_bin_edges = _get_bin_edges(ssp_lgmet, LGMET_LO, LGMET_HI)
    lgmet_weights = _get_triweights_singlepoint(lgmet, lgmet_scatter, lgmet_bin_edges)

    return age_weights, lgmet_weights


@jjit
def _calc_sed_kern(
    t_obs,
    ssp_lgmet,
    ssp_lg_age_gyr,
    ssp_flux,
    t_table,
    logsm_table,
    lgmet,
    lgmet_scatter,
):
    n_met, n_ages, n_wave = ssp_flux.shape
    lgt_table = jnp.log10(t_table)
    _res = _calc_age_met_weights_from_sfh_table(
        t_obs,
        ssp_lgmet,
        ssp_lg_age_gyr,
        lgt_table,
        logsm_table,
        lgmet,
        lgmet_scatter,
    )
    age_weights, lgmet_weights = _res

    age_weights = age_weights.reshape((1, n_ages))
    lgmet_weights = lgmet_weights.reshape((n_met, 1))
    ssp_weights = age_weights * lgmet_weights
    ssp_weights = ssp_weights.reshape((n_met, n_ages, 1))
    sed = jnp.sum(ssp_flux * ssp_weights, axis=(0, 1))
    mstar_obs = 10 ** jnp.interp(jnp.log10(t_obs), lgt_table, logsm_table)
    return sed * mstar_obs


_a = (*[None] * 5, 0, 0, 0)
_calc_sed_vmap = jjit(vmap(_calc_sed_kern, in_axes=_a))


def compute_sed_galpop(
    t_obs,
    ssp_lgmet,
    ssp_lg_age_gyr,
    ssp_flux,
    t_table,
    sfh_table,
    lgmet_params,
):
    """Calculate the SEDs of a galaxy population.

    Parameters
    ----------
    t_obs : float
        Time of observation in Gyr

    ssp_lgmet : ndarray of shape (n_met, )
        SSP bins of log10(Z)

    ssp_lg_age_gyr : ndarray of shape (n_ages, )
        SSP bins of log10(age) in gyr

    ssp_flux : ndarray of shape (n_met, n_ages, n_wave)
        Array storing SSP luminosity in Lsun/Hz

    t_table : ndarray of shape (n_times, )
        Age of the universe in Gyr

    sfh_table : ndarray of shape (n_gals, n_times)
        SFR history for each galaxy in Msun/yr

    lgmet_params : ndarray of shape (n_gals, 2)
        Median metallicity and log-normal scatter for each galaxy

    Returns
    -------
    sed_galpop : ndarray of shape (n_gals, n_wave)
        SED of each galaxy in in Lsun/Hz

    """
    dt_table = _jax_get_dt_array(t_table)
    smh = 1e9 * np.cumsum(sfh_table * dt_table, axis=1)
    smh = np.where(smh == 0, 1, smh)
    logsm_table = np.log10(smh)
    lgmet = lgmet_params[:, 0]
    lgmet_scatter = lgmet_params[:, 1]
    sed_galpop = _calc_sed_vmap(
        t_obs,
        ssp_lgmet,
        ssp_lg_age_gyr,
        ssp_flux,
        t_table,
        logsm_table,
        lgmet,
        lgmet_scatter,
    )
    return sed_galpop, logsm_table
