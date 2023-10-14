"""
"""
from diffstar.defaults import SFR_MIN
from dsps.dust.att_curves import _frac_transmission_from_k_lambda, sbl18_k_lambda
from dsps.photometry.photometry_kernels import calc_obs_mag, calc_rest_mag
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

_interp_vmap = jjit(vmap(jnp.interp, in_axes=[0, None, 0]))


@jjit
def _calc_logmstar_formed(sfh, dt_gyr):
    sfh = jnp.where(sfh < SFR_MIN, SFR_MIN, sfh)
    smh = jnp.cumsum(sfh * dt_gyr) * 1e9
    logsmh = jnp.log10(smh)
    return logsmh


_smh = [0, None]
_calc_logmstar_formed_vmap = jjit(vmap(_calc_logmstar_formed, in_axes=_smh))


@jjit
def _calc_age_met_weights_from_sfh_table_vmap():
    pass


@jjit
def _mult(w1, w2):
    return w1 * w2


_mult_vmap = jjit(vmap(vmap(_mult, in_axes=[None, 0]), in_axes=[0, None]))
_get_weight_matrix = jjit(vmap(_mult_vmap, in_axes=[0, 0]))


@jjit
def interpolate_ssp_photmag_table(z_gals, z_table, ssp_photmag_table):
    iz_hi = jnp.searchsorted(z_table, z_gals)
    iz_lo = iz_hi - 1
    z_lo = z_table[iz_lo]
    z_hi = z_table[iz_hi]
    dz_bin = z_hi - z_lo
    dz = z_gals - z_lo
    w_lo = 1 - (dz / dz_bin)

    ssp_table_zlo = ssp_photmag_table[iz_lo]
    ssp_table_zhi = ssp_photmag_table[iz_hi]

    s = ssp_table_zlo.shape
    outshape = [s[0], *[1 for x in s[1:]]]
    w_lo = w_lo.reshape(outshape)

    gal_photmags = w_lo * ssp_table_zlo + (1 - w_lo) * ssp_table_zhi
    return gal_photmags


_z = [*[None] * 4, 0, *[None] * 4]
_f = [None, None, 0, 0, None, *[None] * 4]
_ssp = [None, 0, *[None] * 7]
_calc_obs_mag_vmap_f = jjit(vmap(calc_obs_mag, in_axes=_f))
_calc_obs_mag_vmap_f_ssp = jjit(
    vmap(vmap(_calc_obs_mag_vmap_f, in_axes=_ssp), in_axes=_ssp)
)
_calc_obs_mag_vmap_f_ssp_z = jjit(vmap(_calc_obs_mag_vmap_f_ssp, in_axes=_z))

_calc_obs_mag_vmap_f_ssp_singlemet = jjit(vmap(_calc_obs_mag_vmap_f, in_axes=_ssp))
_calc_obs_mag_vmap_f_ssp_z_singlemet = jjit(
    vmap(_calc_obs_mag_vmap_f_ssp_singlemet, in_axes=_z)
)


_calc_rest_mag_vmap_f = jjit(vmap(calc_rest_mag, in_axes=[None, None, 0, 0]))
_calc_rest_mag_vmap_f_ssp = jjit(
    vmap(
        vmap(_calc_rest_mag_vmap_f, in_axes=[None, 0, None, None]),
        in_axes=[None, 0, None, None],
    )
)

_calc_rest_mag_vmap_f_ssp_singlemet = jjit(
    vmap(_calc_rest_mag_vmap_f, in_axes=[None, 0, None, None]),
)


@jjit
def _get_filter_effective_wavelength_rest(filter_wave, filter_trans):
    norm = jnp.trapz(filter_trans, x=filter_wave)
    lambda_eff = jnp.trapz(filter_trans * filter_wave, x=filter_wave) / norm
    return lambda_eff


@jjit
def _get_filter_lambda_eff_obsframe_kern(filter_wave, filter_trans, redshift):
    lambda_eff_rest = _get_filter_effective_wavelength_rest(filter_wave, filter_trans)
    lambda_eff = lambda_eff_rest / (1 + redshift)
    return lambda_eff


@jjit
def _get_effective_attenuation(filter_wave, filter_trans, redshift, dust_params):
    """Attenuation factor at the effective wavelength of the filter"""

    lambda_eff = _get_filter_lambda_eff_obsframe_kern(
        filter_wave, filter_trans, redshift
    )
    lambda_eff_micron = lambda_eff / 10_000

    uv_bump_ampl, plaw_slope, dust_Av = dust_params
    k_lambda = sbl18_k_lambda(lambda_eff_micron, uv_bump_ampl, plaw_slope)
    frac_transmission = _frac_transmission_from_k_lambda(k_lambda, dust_Av)
    return frac_transmission


_g = in_axes = [None, None, 0, 0]
_f = in_axes = [0, 0, None, None]
_get_effective_attenuation_vmap = jjit(vmap(vmap(_get_effective_attenuation, _f), _g))
