"""
"""
from dsps.photometry.photometry_kernels import calc_obs_mag, calc_rest_mag
from jax import jit as jjit
from jax import vmap

from ..defaults import DEFAULT_COSMO_PARAMS
from ..sed.sed_kernels_singlemet import calc_rest_sed_singlegal

_F = (*[None] * 2, 0, 0, *[None] * 5)
calc_obs_mag_vmap = jjit(vmap(calc_obs_mag, in_axes=_F))

_R = (None, None, 0, 0)
calc_rest_mag_vmap = jjit(vmap(calc_rest_mag, in_axes=_R))


@jjit
def calc_photometry_singlegal(
    z_obs,
    mah_params,
    ms_params,
    q_params,
    ssp_data,
    diffskypop_params,
    rest_filter_waves,
    rest_filter_trans,
    obs_filter_waves,
    obs_filter_trans,
    cosmo_params=DEFAULT_COSMO_PARAMS,
):
    """Calculate the photometry of an individual diffsky galaxy"""

    _res = calc_rest_sed_singlegal(
        z_obs,
        mah_params,
        ms_params,
        q_params,
        ssp_data,
        diffskypop_params,
        cosmo_params=cosmo_params,
    )
    rest_sed, rest_sed_nodust = _res[:2]

    # Calculate mags including dust attenuation
    obs_mags = calc_obs_mag_vmap(
        ssp_data.ssp_wave,
        rest_sed,
        obs_filter_waves,
        obs_filter_trans,
        z_obs,
        *cosmo_params[:-1],
    )

    rest_mags = calc_rest_mag(
        ssp_data.ssp_wave, rest_sed, rest_filter_waves, rest_filter_trans
    )

    # Calculate mags excluding dust attenuation
    obs_mags_nodust = calc_obs_mag_vmap(
        ssp_data.ssp_wave,
        rest_sed_nodust,
        obs_filter_waves,
        obs_filter_trans,
        z_obs,
        *cosmo_params[:-1],
    )

    rest_mags_nodust = calc_rest_mag(
        ssp_data.ssp_wave, rest_sed_nodust, rest_filter_waves, rest_filter_trans
    )

    return rest_mags, obs_mags, rest_mags_nodust, obs_mags_nodust


_P = (*[0] * 4, *[None] * 7)
_calc_photometry_galpop_kern = jjit(vmap(calc_photometry_singlegal, in_axes=_P))


@jjit
def calc_photometry_galpop(
    z_obs,
    mah_params,
    ms_params,
    q_params,
    ssp_data,
    diffskypop_params,
    rest_filter_waves,
    rest_filter_trans,
    obs_filter_waves,
    obs_filter_trans,
    cosmo_params=DEFAULT_COSMO_PARAMS,
):
    """Compute SED and photometry for population of Diffsky galaxies

    Parameters
    ----------
    z_obs : ndarray, shape (n_gals, )
        Redshift of each galaxy

    mah_params : ndarray, shape (n_gals, 4)
        Diffmah params specifying the mass assembly of the dark matter halo
        diffmah_params = (logm0, logtc, early_index, late_index)

    ms_params : ndarray, shape (n_gals, 5)
        Diffstar params for the star-formation effiency and gas consumption timescale
        ms_params = (lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep)

    q_params : ndarray, shape (n_gals, 4)
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

    cosmo_params : ndarray, shape (n_cosmo, )
        Defined in lsstdesc_diffsky.defaults, cosmo_params = (Om0, w0, wa, h, fb)

    Returns
    ----------
    rest_mags : ndarray, shape (n_gals, n_rest_filters)

    obs_mags : ndarray, shape (n_gals, n_obs_filters)

    rest_mags_nodust : ndarray, shape (n_gals, n_rest_filters)

    obs_mags_nodust : ndarray, shape (n_gals, n_obs_filters)

    """
    return _calc_photometry_galpop_kern(
        z_obs,
        mah_params,
        ms_params,
        q_params,
        ssp_data,
        diffskypop_params,
        rest_filter_waves,
        rest_filter_trans,
        obs_filter_waves,
        obs_filter_trans,
        cosmo_params,
    )
