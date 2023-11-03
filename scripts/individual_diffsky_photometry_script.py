"""
"""
import argparse
import os

import numpy as np
from dsps.data_loaders import load_transmission_curve
from dsps.data_loaders.defaults import TransmissionCurve
from galsim import roman

from lsstdesc_diffsky import read_diffskypop_params
from lsstdesc_diffsky.io_utils.load_diffsky_healpixel import (
    get_healpixel_bname_from_ra_dec_z,
    load_diffsky_params,
    load_healpixel,
)
from lsstdesc_diffsky.photometry.photometry_kernels_singlemet import (
    calc_photometry_singlegal,
)
from lsstdesc_diffsky.photometry.precompute_ssp_tables import (
    interpolate_filter_trans_curves,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("drn", help="directory storing diffsky healpixels")
    parser.add_argument("ra", help="ra coordinate of diffsky galaxy")
    parser.add_argument("dec", help="dec coordinate of diffsky galaxy")
    parser.add_argument("z", help="redshift of diffsky galaxy")
    parser.add_argument("galid", help="ID of diffsky galaxy")
    args = parser.parse_args()

    drn = args.drn
    ra = args.ra
    dec = args.dec
    z = args.z
    galid = args.galid

    bname = get_healpixel_bname_from_ra_dec_z(ra, dec, z)
    fname = os.path.join(drn, bname)

    mock, metadata = load_healpixel(fname)
    msk = mock["galaxy_id"] == galid
    assert msk.sum() == 1, "No galaxy in healpixel with ID = {0}".format(galid)
    mock = mock[msk]

    lsst_keys_to_print = [key for key in mock.keys() if "LSST" in key.lower()]
    roman_keys_to_print = [key for key in mock.keys() if "Roman" in key.lower()]

    diffsky_params = load_diffsky_params(mock)

    z_obs = float(mock["redshift"][0])
    zerr_pat = "Input redshift={0} but redshift of galaxy with ID={1} is {2}"
    zerr_msg = zerr_pat.format(z, galid, z_obs)
    assert np.allclose(z_obs, z, zerr_msg, atol=0.01)

    ra_galid = float(mock["ra"][0])
    ra_pat = "Input ra={0} but ra of galaxy with ID={1} is {2}"
    ra_msg = ra_pat.format(ra, galid, ra_galid)
    assert np.allclose(ra_galid, ra, ra_msg, atol=0.01)

    dec_galid = float(mock["dec"][0])
    dec_pat = "Input dec={0} but dec of galaxy with ID={1} is {2}"
    dec_msg = dec_pat.format(dec, galid, dec_galid)
    assert np.allclose(dec_galid, dec, dec_msg, atol=0.01)

    diffskypop_params = read_diffskypop_params("roman_rubin_2023")

    roman_filter_list = ("R062", "Z087", "Y106", "J129", "W146", "H158", "F184", "K213")
    rbps = roman.getBandpasses()

    roman_tcurves = [
        TransmissionCurve(
            rbps[band].wave_list * 10,
            rbps[band](rbps[band].wave_list),
        )
        for band in roman_filter_list
    ]

    calc_photometry_singlegal(
        z_obs,
        diffsky_params.mah_params,
        diffsky_params.ms_params,
        diffsky_params.q_params,
        ssp_data,
        diffskypop_params,
        rest_filter_waves,
        rest_filter_trans,
        obs_filter_waves,
        obs_filter_trans,
    )
