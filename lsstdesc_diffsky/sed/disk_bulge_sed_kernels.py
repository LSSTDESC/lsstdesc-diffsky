"""
"""
from diffsky.experimental.dspspop.boris_dust import _get_funo_from_u_params_singlegal
from diffsky.experimental.dspspop.burstshapepop import (
    _get_burstshape_galpop_from_u_params,
)
from diffsky.experimental.dspspop.dust_deltapop import (
    _get_dust_delta_galpop_from_u_params,
)
from diffsky.experimental.dspspop.lgavpop import _get_lgav_galpop_from_u_params
from diffsky.experimental.dspspop.lgfburstpop import _get_lgfburst_galpop_from_u_params
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
    _age_weights_from_params as _burst_age_weights_from_params,
)
from dsps.experimental.diffburst import (
    _age_weights_from_u_params as _burst_age_weights_from_u_params,
)
from dsps.experimental.diffburst import (
    _get_params_from_u_params as _get_diffburst_params_from_u_params,
)
from dsps.sed.metallicity_weights import calc_lgmet_weights_from_lognormal_mdf
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

from ..defaults import DEFAULT_COSMO_PARAMS
from ..disk_bulge_modeling.disk_bulge_kernels import (
    _decompose_sfh_singlegal_into_bulge_disk_knots,
)
from .sed_kernels import _get_galprops_at_t_obs_singlegal

_T = (None, None, 0)
_frac_transmission_from_k_lambda_age_vmap = jjit(
    vmap(_frac_transmission_from_k_lambda, in_axes=_T)
)


def calc_rest_sed_disk_bulge_knot_singlegal(
    z_obs,
    diffmah_params,
    diffstar_ms_params,
    diffstar_q_params,
    fbulge_params,
    fknot,
    ssp_lgmet,
    ssp_lg_age_gyr,
    ssp_wave_ang,
    ssp_flux,
    lgfburst_pop_u_params,
    burstshapepop_u_params,
    lgav_pop_u_params,
    dust_delta_pop_u_params,
    fracuno_pop_u_params,
    met_params,
    cosmo_params=DEFAULT_COSMO_PARAMS,
):
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

    mzr_params, lgmet_scatter = met_params[:-1], met_params[-1]
    _galprops_at_t_obs = _get_galprops_at_t_obs_singlegal(
        t_obs, t_table, sfh_table, mzr_params, ssp_lg_age_gyr
    )
    logsm_t_obs, logssfr_t_obs, lgmet_t_obs, smooth_age_weights = _galprops_at_t_obs[:4]
    mstar_t_obs = 10**logsm_t_obs

    lgmet_weights = calc_lgmet_weights_from_lognormal_mdf(
        lgmet_t_obs, lgmet_scatter, ssp_lgmet
    )

    # Compute burst fraction and burst shape
    lgfburst = _get_lgfburst_galpop_from_u_params(
        logsm_t_obs, logssfr_t_obs, lgfburst_pop_u_params
    )
    fburst = 10**lgfburst

    diffburst_u_params = _get_burstshape_galpop_from_u_params(
        logsm_t_obs, logssfr_t_obs, burstshapepop_u_params
    )
    burstshape_params = jnp.array(
        _get_diffburst_params_from_u_params(diffburst_u_params)
    )
    ssp_lg_age_yr = ssp_lg_age_gyr + 9
    age_weights_singleburst = _burst_age_weights_from_params(
        ssp_lg_age_yr, burstshape_params
    )

    # Decompose SFH into disk/bulge/knots
    _res = _decompose_sfh_singlegal_into_bulge_disk_knots(
        fbulge_params,
        fknot,
        t_obs,
        t_table,
        sfh_table,
        fburst,
        age_weights_singleburst,
        ssp_lg_age_gyr,
    )
    mbulge, mdd, mknot, mburst = _res[:4]
    bulge_age_weights, dd_age_weights, knot_age_weights = _res[4:7]
    bulge_sfh, gal_frac_bulge_t_obs = _res[7:]

    # Compute SED of each component, neglecting dust attenuation
    n_met, n_age, n_wave = ssp_flux.shape
    lgmet_weights = lgmet_weights.reshape((n_met, 1, 1))
    bulge_weights = lgmet_weights * bulge_age_weights.reshape((1, n_age, 1))
    dd_weights = lgmet_weights * dd_age_weights.reshape((1, n_age, 1))
    knot_weights = lgmet_weights * knot_age_weights.reshape((1, n_age, 1))

    rest_sed_bulge_nodust = jnp.sum(bulge_weights * ssp_flux, axis=(0, 1)) * mbulge
    rest_sed_dd_nodust = jnp.sum(dd_weights * ssp_flux, axis=(0, 1)) * mdd
    rest_sed_knot_nodust = jnp.sum(knot_weights * ssp_flux, axis=(0, 1)) * mknot

    # Compute transmission curve (assumed same for all components)
    lgav = _get_lgav_galpop_from_u_params(logsm_t_obs, logssfr_t_obs, lgav_pop_u_params)
    dust_delta = _get_dust_delta_galpop_from_u_params(
        logsm_t_obs, logssfr_t_obs, dust_delta_pop_u_params
    )
    frac_unobscured = _get_funo_from_u_params_singlegal(
        logsm_t_obs, lgfburst, logssfr_t_obs, ssp_lg_age_gyr, fracuno_pop_u_params
    )

    ssp_wave_micron = ssp_wave_ang / 1e4
    dust_Av = 10**lgav
    dust_Eb = _get_eb_from_delta(dust_delta)
    k_lambda = sbl18_k_lambda(
        ssp_wave_micron, UV_BUMP_W0, UV_BUMP_DW, dust_Eb, dust_delta
    )
    frac_dust_trans = _frac_transmission_from_k_lambda_age_vmap(
        k_lambda, dust_Av, frac_unobscured
    )
    frac_dust_trans = frac_dust_trans.reshape((1, n_age, n_wave))

    rest_sed_bulge = (
        jnp.sum(bulge_weights * ssp_flux * frac_dust_trans, axis=(0, 1)) * mbulge
    )

    rest_sed_dd = jnp.sum(dd_weights * ssp_flux * frac_dust_trans, axis=(0, 1)) * mdd
    rest_sed_knot = (
        jnp.sum(knot_weights * ssp_flux * frac_dust_trans, axis=(0, 1)) * mknot
    )

    rest_seds_nodust = rest_sed_bulge_nodust, rest_sed_dd_nodust, rest_sed_knot_nodust
    rest_seds = rest_sed_bulge, rest_sed_dd, rest_sed_knot
    masses = mbulge, mdd, mknot, mburst, mstar_t_obs

    ret = (*rest_seds, *rest_seds_nodust, *masses, gal_frac_bulge_t_obs)
    return ret
