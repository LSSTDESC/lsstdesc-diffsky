"""
"""
import numpy as np
from diffsky.experimental.dspspop.burstshapepop import (
    _get_burstshape_galpop_from_params,
)
from diffsky.experimental.dspspop.dustpop import _frac_dust_transmission_kernel
from diffsky.experimental.dspspop.lgfburstpop import _get_lgfburst_galpop_from_u_params
from dsps.experimental.diffburst import _age_weights_from_u_params
from dsps.metallicity.mzr import mzr_model
from dsps.sed import calc_ssp_weights_sfh_table_lognormal_mdf
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from .photometry.get_SFH_from_params import (
    get_log_safe_ssfr,
    get_logsm_sfr_obs,
    get_sfh_from_params,
)

_A = (None, 0)
_age_weights_from_u_params_vmap = jjit(vmap(_age_weights_from_u_params, in_axes=_A))

_g = (None, 0, 0, None, None, None, None)
calc_ssp_weights_sfh_table_lognormal_mdf_vmap = jjit(
    vmap(calc_ssp_weights_sfh_table_lognormal_mdf, in_axes=_g)
)


def generate_seds_and_photometry(
    t_table,
    mah_params,
    u_ms_params,
    u_q_params,
    cosmology,
    redshift,
    met_params,
    ssp_lgmet,
    ssp_lg_age_gyr,
    filter_waves_obsmags,
    filter_trans_obsmags,
    lgfburst_pop_u_params,
    burstshapepop_u_params,
    lgav_u_params,
    dust_delta_u_params,
    fracuno_pop_u_params,
    lg_met_scatter=0.25,
):
    """
    Compute SEDs and photometry from model parameters for each galaxy
    """
    n_gals = mah_params.shape[0]
    n_age = ssp_lg_age_gyr.shape[0]
    n_met = ssp_lgmet.shape[0]

    t_obs = cosmology.age(redshift).value
    lgt0 = np.log10(cosmology.age(0).value)

    return_dict = dict()

    # get SFH table and observed stellar mass
    sfh_table = get_sfh_from_params(mah_params, u_ms_params, u_q_params, lgt0, t_table)
    logsm_obs, sfr_obs = get_logsm_sfr_obs(sfh_table, t_obs, t_table)
    log_ssfr = get_log_safe_ssfr(logsm_obs, sfr_obs)

    return_dict["logsm_obs"] = logsm_obs
    return_dict["sfr"] = sfr_obs
    return_dict["log_ssfr"] = log_ssfr

    # generate metallicities
    lg_met_mean = mzr_model(logsm_obs, t_obs, *met_params)

    return_dict["lg_met_mean"] = lg_met_mean
    return_dict["lg_met_scatter"] = np.zeros(n_gals) + lg_met_scatter

    _res = calc_ssp_weights_sfh_table_lognormal_mdf_vmap(
        t_table,
        sfh_table,
        return_dict["lg_met_mean"],
        return_dict["lg_met_scatter"],
        ssp_lgmet,
        ssp_lg_age_gyr,
        t_obs,
    )
    lgmet_weights, smooth_age_weights = _res[1:]

    gal_lgf_burst = _get_lgfburst_galpop_from_u_params(
        logsm_obs, log_ssfr, lgfburst_pop_u_params
    )
    gal_fburst = 10**gal_lgf_burst

    burstshape_u_params = _get_burstshape_galpop_from_params(
        logsm_obs, log_ssfr, burstshapepop_u_params
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
    weights = _w / _norm.reshape((n_gals, 1, 1))

    dummy_dust_key = 0
    _res = _frac_dust_transmission_kernel(
        dummy_dust_key,
        redshift,
        logsm_obs,
        log_ssfr,
        gal_lgf_burst,
        ssp_lg_age_gyr,
        filter_waves_obsmags,
        filter_trans_obsmags,
        lgav_u_params,
        dust_delta_u_params,
        fracuno_pop_u_params,
    )
    gal_frac_trans, gal_att_curve_params, gal_frac_unobs = _res

    return_dict["dust_Eb"] = gal_att_curve_params[:, 0]
    return_dict["dust_delta"] = gal_att_curve_params[:, 1]
    return_dict["dust_Av"] = gal_att_curve_params[:, 2]
    return_dict["attenuation_factors"] = gal_frac_trans
    return_dict["frac_unobscured"] = gal_frac_unobs

    mags, seds = get_mags_seds()

    return_dict["SEDs"] = seds
    for col in mags.colnames:
        return_dict[col] = mags[col]

    return return_dict
