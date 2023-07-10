import matplotlib.pyplot as plt
import astropy.constants
from astropy.constants import iaudata
from astropy import units as u
import numpy as np
import pandas as pd
import galsim
import GCRCatalogs
from load_sps_data import load_ssp_spectra
from skysim_calc_sed import (_calc_rest_sed_single_diffstar_gal,
                             _get_att_curve_kern)
from lsstdesc_diffsky.constants import MAH_PNAMES, MS_U_PNAMES, Q_U_PNAMES

DUST_PNAMES = ['dust_Eb', 'dust_delta', 'dust_Av']


class DspsGalaxySedFactory:
    """Factory class to produce galaxy SEDs from the DSPS model."""
    def __init__(self, DSPS_data_dir, cosmology):
        self.ssp_data = load_ssp_spectra(DSPS_data_dir)
        self.cosmology = cosmology

    def create(self, galaxy_pars):
        """Return the rest-frame SED of the galaxy."""
        zz = galaxy_pars['redshift_true']
        t_obs = gc.cosmology.lookback_time(zz)
        sed_args = (t_obs,
                    self.ssp_data.lgZsun_bin_mids,
                    self.ssp_data.log_age_gyr,
                    self.ssp_data.ssp_spectra,
                    np.array([galaxy_pars[_] for _ in MAH_PNAMES]),
                    np.array([galaxy_pars[_] for _ in MS_U_PNAMES]),
                    np.array([galaxy_pars[_] for _ in Q_U_PNAMES]))
        dust_params = np.array([galaxy_pars[_] for _ in DUST_PNAMES])

        rest_sed, galaxy_data = _calc_rest_sed_single_diffstar_gal(*sed_args)
        atten = _get_att_curve_kern(self.ssp_data.ssp_wave, dust_params)
        Lnu = np.array(rest_sed*atten)  # Lsun/Hz

        # Collect conversion factors = L_sun*ergs_per_joule*clight/4./pi/dl**2
        ERG_PER_JOULE = 1e7
        CM_PER_MPC = 3.085677581491367e+24
        clight = astropy.constants.c.value*1e2  # cm/s
        dl = self.cosmology.luminosity_distance(zz).value*CM_PER_MPC
        factor = iaudata.L_sun.value*ERG_PER_JOULE*clight/4/np.pi/dl**2

        wl = self.ssp_data.ssp_wave*1e-8  # convert from Angstroms to cm
        Flambda = Lnu*factor/wl**2
        lut = galsim.LookupTable(self.ssp_data.ssp_wave, Flambda)
        return galsim.SED(lut, wave_type=u.Angstrom, flux_type='flambda')


if __name__ == '__main__':
    DSPS_data_dir = '/global/cfs/cdirs/descssim/imSim/DSPS/ssp_data'

    GCRCatalogs.set_root_dir_by_site('nersc')
    gc = GCRCatalogs.load_catalog('skysim_v3.1.0_small')

    sed_factory = DspsGalaxySedFactory(DSPS_data_dir, gc.cosmology)

    columns = (['galaxy_id', 'ra', 'dec', 'redshift_true', 'mag_i_lsst']
               + MAH_PNAMES + MS_U_PNAMES + Q_U_PNAMES + DUST_PNAMES)

    # Grab the first chunk of SkySim galaxy data.
    for i, chunk in enumerate(gc.get_quantities(columns, return_iterator=True)):
        df = pd.DataFrame(chunk)
        break

    plt.figure()
    for _, galaxy_pars in df.head(10).iterrows():
        sed = sed_factory.create(galaxy_pars)
        sed_values = sed(sed.wave_list)
        plt.plot(sed.wave_list, sed_values)
    plt.yscale('log')
    plt.xlim(1, 1100)
    plt.ylim(1e-3, 1e4)
    plt.xlabel('wavelength (nm)')
    plt.ylabel('photons/nm/cm^2/s')
    plt.savefig('galaxy_seds.png')
