"""
"""
import numpy as np

from . import photometry_interpolation_kernels as pik


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


def precompute_ssp_obsmags_on_z_table_singlemet(
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

    ssp_fluxes : array of shape (n_age, n_spec)

    filter_waves : array of shape (n_filters, n_trans_curve)

    filter_trans : array of shape (n_filters, n_trans_curve)

    z_table : array of shape (n_redshift, )

    Om0 : float

    w0 : float

    wa : float

    h : float

    Returns
    -------
    ssp_photmag_table : array of shape (n_redshift, n_age, n_filters)

    """
    ssp_obsmag_table = pik._calc_obs_mag_vmap_f_ssp_z_singlemet(
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


def precompute_ssp_restmags_singlemet(ssp_wave, ssp_fluxes, filter_waves, filter_trans):
    """Precompute restframe magnitudes of a collection of SEDs

    Parameters
    ----------
    ssp_wave : array of shape (n_spec, )

    ssp_fluxes : array of shape (n_age, n_spec)

    filter_waves : array of shape (n_filters, n_trans_curve)

    filter_trans : array of shape (n_filters, n_trans_curve)

    Returns
    -------
    ssp_photmag_table : array of shape (n_age, n_filters)

    """
    ssp_restmag_table = pik._calc_rest_mag_vmap_f_ssp_singlemet(
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
