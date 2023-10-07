"""
"""
from dsps.photometry.photometry_kernels import calc_obs_mag, calc_rest_mag
from jax import jit as jjit
from jax import vmap

from ..defaults import DEFAULT_COSMO_PARAMS
from ..sed import calc_rest_sed_singlegal

_F = (*[None] * 2, 0, 0, *[None] * 5)
calc_obs_mag_vmap = jjit(vmap(calc_obs_mag, in_axes=_F))

_R = (None, None, 0, 0)
calc_rest_mag_vmap = jjit(vmap(calc_rest_mag, in_axes=_R))


@jjit
def calc_photometry_singlegal(
    z_obs,
    diffmah_params,
    diffstar_ms_params,
    diffstar_q_params,
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
    rest_filter_waves,
    rest_filter_trans,
    obs_filter_waves,
    obs_filter_trans,
    cosmo_params=DEFAULT_COSMO_PARAMS,
):
    """Calculate the photometry of an individual diffsky galaxy"""
    _res = calc_rest_sed_singlegal(
        z_obs,
        diffmah_params,
        diffstar_ms_params,
        diffstar_q_params,
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
        cosmo_params=cosmo_params,
    )
    rest_sed, rest_sed_nodust = _res[:2]

    # Calculate mags including dust attenuation
    obs_mags = calc_obs_mag_vmap(
        ssp_wave_ang,
        rest_sed,
        obs_filter_waves,
        obs_filter_trans,
        z_obs,
        *cosmo_params[:-1],
    )

    rest_mags = calc_rest_mag(
        ssp_wave_ang, rest_sed, rest_filter_waves, rest_filter_trans
    )

    # Calculate mags excluding dust attenuation
    obs_mags_nodust = calc_obs_mag_vmap(
        ssp_wave_ang,
        rest_sed_nodust,
        obs_filter_waves,
        obs_filter_trans,
        z_obs,
        *cosmo_params[:-1],
    )

    rest_mags_nodust = calc_rest_mag(
        ssp_wave_ang, rest_sed_nodust, rest_filter_waves, rest_filter_trans
    )

    return rest_mags, obs_mags, rest_mags_nodust, obs_mags_nodust


_P = (*[0] * 4, *[None] * 15)
_calc_photometry_galpop_kern = jjit(vmap(calc_photometry_singlegal, in_axes=_P))


@jjit
def calc_photometry_galpop(
    z_obs,
    diffmah_params,
    diffstar_ms_params,
    diffstar_q_params,
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
    rest_filter_waves,
    rest_filter_trans,
    obs_filter_waves,
    obs_filter_trans,
    cosmo_params=DEFAULT_COSMO_PARAMS,
):
    return _calc_photometry_galpop_kern(
        z_obs,
        diffmah_params,
        diffstar_ms_params,
        diffstar_q_params,
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
        rest_filter_waves,
        rest_filter_trans,
        obs_filter_waves,
        obs_filter_trans,
        cosmo_params,
    )
