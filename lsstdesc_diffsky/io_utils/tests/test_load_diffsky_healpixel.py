"""
"""
import numpy as np

from ...constants import (
    BURSTSHAPE_PNAMES,
    FBULGE_PNAMES,
    MAH_PNAMES,
    MS_PNAMES,
    Q_PNAMES,
)
from ..load_diffsky_healpixel import (
    get_healpixel_bname_from_ra_dec_z,
    load_diffsky_params,
)


def test_load_diffsky_params():
    fake_data = dict()
    ngals = 100
    ptypes = (BURSTSHAPE_PNAMES, FBULGE_PNAMES, MAH_PNAMES, MS_PNAMES, Q_PNAMES)
    for ptype in ptypes:
        for key in ptype:
            fake_data[key] = np.zeros(ngals)
    fake_data["fburst"] = np.zeros(ngals)
    fake_data["fknot"] = np.zeros(ngals)

    diffsky_params = load_diffsky_params(fake_data)
    assert diffsky_params.mah_params.shape == (ngals, 4)
    assert diffsky_params.ms_params.shape == (ngals, 5)
    assert diffsky_params.q_params.shape == (ngals, 4)
    assert diffsky_params.fburst.shape == (ngals,)
    assert diffsky_params.burstshape_params.shape == (ngals, 2)
    assert diffsky_params.fbulge_params.shape == (ngals, 3)
    assert diffsky_params.fknot.shape == (ngals,)


def test_get_healpixel_bname_from_ra_dec_z():
    ra, dec = 9.59383, -43.7799

    redshift = 0.5
    bname = get_healpixel_bname_from_ra_dec_z(ra, dec, redshift)
    assert bname == "roman_rubin_2023_z_0_1_cutout_10307.hdf5"

    redshift = 1.5
    bname = get_healpixel_bname_from_ra_dec_z(ra, dec, redshift)
    assert bname == "roman_rubin_2023_z_1_2_cutout_10307.hdf5"

    redshift = 2.5
    bname = get_healpixel_bname_from_ra_dec_z(ra, dec, redshift)
    assert bname == "roman_rubin_2023_z_2_3_cutout_10307.hdf5"
