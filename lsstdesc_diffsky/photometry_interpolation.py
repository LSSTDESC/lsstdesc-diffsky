"""
"""
import numpy as np
from jax import numpy as jnp
from . import photometry_interpolation_kernels as pik


def get_interpolated_photometry(
    ssp_z_table,
    ssp_restmag_table,
    ssp_obsmag_table,
    ssp_lgZsun_bin_mids,
    ssp_log_ages,
    gal_lgt_table,
    gal_z_obs,
    gal_t_obs,
    gal_logsm_obs,
    gal_logsm_table,
    gal_lgmet,
    gal_lgmet_scatter,
    lgt0,
    attenuation_factors=None,
):
    """Calculate restframe and observed photometry of galaxies in a lightcone

    Method is to interpolate precomputed photometry of SSP SEDs

    Parameters
    ----------
    ssp_z_table : array of shape (n_z_table_ssp, )
        Array must be monotonically increasing

    ssp_restmag_table : array of shape (n_z_table_ssp, n_met, n_age, n_rest_filters, )

    ssp_obsmag_table : array of shape (n_z_table_ssp, n_met, n_age, n_obs_filters,)

    ssp_lgZsun_bin_mids : array of shape (n_met, )

    ssp_log_ages : array of shape (n_ages, )

    gal_lgt_table : array of shape (n_t_table_gals, )

    gal_z_obs : array of shape (n_gals, )

    gal_t_obs : array of shape (n_gals, )

    gal_logsm_obs : array of shape (n_gals, )

    gal_logsm_table : array of shape (n_gals, n_t_table_gals)

    gal_lgmet : array of shape (n_gals, )

    gal_lgmet_scatter : array of shape (n_gals, )

    lgt0 : float

    attenuation_factors : array of shape (n_gals, n_filters), optional
        Fraction of the flux in each band that is transmitted through the dust
        Default is None, for zero attenuation

    Returns
    -------
    gal_obsmags : array of shape (n_gals, n_obs_filters)

    gal_restmags : array of shape (n_gals, n_rest_filters)

    """
    msg = "ssp_z_table must be monotonically increasing"
    assert jnp.all(jnp.diff(ssp_z_table) > 0), msg

    age_weights, lgmet_weights = pik._calc_age_met_weights_from_sfh_table_vmap(
        gal_t_obs,
        ssp_lgZsun_bin_mids,
        ssp_log_ages,
        gal_lgt_table,
        gal_logsm_table,
        gal_lgmet,
        gal_lgmet_scatter,
    )

    weight_matrix = pik._get_weight_matrix(lgmet_weights, age_weights)

    ssp_obsmag_table_pergal = pik.interpolate_ssp_photmag_table(
        gal_z_obs, ssp_z_table, ssp_obsmag_table
    )
    n_gals, n_met, n_age, n_filters = ssp_obsmag_table_pergal.shape

    _w = weight_matrix.reshape((n_gals, n_met, n_age, 1))
    ssp_obsflux_table_pergal = 10 ** (-0.4 * ssp_obsmag_table_pergal)

    gal_mstar_obs = (10**gal_logsm_obs).reshape((n_gals, 1))
    gal_obsflux = jnp.sum(_w * ssp_obsflux_table_pergal, axis=(1, 2)) * gal_mstar_obs

    if attenuation_factors is not None:
        gal_obsflux = gal_obsflux * attenuation_factors

    gal_obsmags = -2.5 * jnp.log10(gal_obsflux)

    ssp_restmag_table = ssp_restmag_table.reshape((1, n_met, n_age, n_filters))
    ssp_restflux_table = 10 ** (-0.4 * ssp_restmag_table)
    gal_restflux = jnp.sum(_w * ssp_restflux_table, axis=(1, 2)) * gal_mstar_obs

    if attenuation_factors is not None:
        gal_restflux = gal_restflux * attenuation_factors

    gal_restmags = -2.5 * jnp.log10(gal_restflux)

    return gal_obsmags, gal_restmags


def precompute_ssp_obsmags_on_z_table(
    ssp_wave,
    ssp_fluxes,
    filter_waves,
    filter_trans,
    z_table,
    Om0,
    w0,
    wa,
    h,
):
    """Precompute observed magnitudes of a collection of SEDs on a redshift grid

    Parameters
    ----------
    ssp_wave : array of shape (n_spec, )

    ssp_fluxes : array of shape (n_met, n_age, n_spec)

    filter_waves : array of shape (n_filters, n_trans_curve)

    filter_trans : array of shape (n_filters, n_trans_curve)

    z_table : array of shape (n_redshift, )

    Om0 : float

    w0 : float

    wa : float

    h : float

    Returns
    -------
    ssp_photmag_table : array of shape (n_redshift, n_met, n_age, n_filters)

    """
    ssp_obsmag_table = pik._calc_obs_mag_vmap_f_ssp_z(
        ssp_wave, ssp_fluxes, filter_waves, filter_trans, z_table, Om0, w0, wa, h
    )
    return ssp_obsmag_table


def precompute_ssp_restmags(ssp_wave, ssp_fluxes, filter_waves, filter_trans):
    """Precompute restframe magnitudes of a collection of SEDs

    Parameters
    ----------
    ssp_wave : array of shape (n_spec, )

    ssp_fluxes : array of shape (n_met, n_age, n_spec)

    filter_waves : array of shape (n_filters, n_trans_curve)

    filter_trans : array of shape (n_filters, n_trans_curve)

    Returns
    -------
    ssp_photmag_table : array of shape (n_met, n_age, n_filters)

    """
    ssp_restmag_table = pik._calc_rest_mag_vmap_f_ssp(
        ssp_wave, ssp_fluxes, filter_waves, filter_trans
    )
    return ssp_restmag_table


def precompute_dust_attenuation(filter_waves, filter_trans, redshift, dust_params):
    """Precompute the attenuation of each galaxy in each band

    Parameters
    ----------
    filter_waves : array of shape (n_filters, n_trans_curve)

    filter_trans : array of shape (n_filters, n_trans_curve)

    redshift : array of shape (n_gals, )

    dust_params : array of shape (n_gals, 3)
        dust parameters are (Eb, delta, Av)

    Returns
    -------
    attenuation_factors : array of shape (n_gals, n_filters)
        Fraction of the flux in each band that is attenuated by dust

    """
    attenuation_factors = pik._get_effective_attenuation_vmap(
        filter_waves, filter_trans, redshift, dust_params
    )
    return attenuation_factors


def interpolate_filter_trans_curves(wave_filters, trans_filters, n=None):
    """Interpolate a collection of filter transmission curves to a common length.
    Convenience function for analyses vmapping over broadband colors.

    Parameters
    ----------
    wave_filters : sequence of n_filters ndarrays

    trans_filters : sequence of n_filters ndarrays

    n : int, optional
        Desired length of the output transmission curves.
        Default is equal to the smallest length transmission curve

    Returns
    -------
    wave_filters : ndarray of shape (n_filters, n)

    trans_filters : ndarray of shape (n_filters, n)

    """
    wave0 = wave_filters[0]
    wave_min, wave_max = wave0.min(), wave0.max()

    if n is None:
        n = np.min([x.size for x in wave_filters])

    for wave, trans in zip(wave_filters, trans_filters):
        wave_min = min(wave_min, wave.min())
        wave_max = max(wave_max, wave.max())

    wave_collector = []
    trans_collector = []
    for wave, trans in zip(wave_filters, trans_filters):
        wave_min, wave_max = wave.min(), wave.max()
        new_wave = np.linspace(wave_min, wave_max, n)
        new_trans = np.interp(new_wave, wave, trans)
        wave_collector.append(new_wave)
        trans_collector.append(new_trans)
    return np.array(wave_collector), np.array(trans_collector)
