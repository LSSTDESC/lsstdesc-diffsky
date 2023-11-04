"""
"""
import typing
from collections import OrderedDict

import h5py
import healpy as hp
import numpy as np

from ..constants import (
    BURSTSHAPE_PNAMES,
    FBULGE_PNAMES,
    MAH_PNAMES,
    MS_PNAMES,
    Q_PNAMES,
)
from ..defaults import ROMAN_RUBIN_2023_HPIX_PAT, ROMAN_RUBIN_2023_NSIDE

ALL_DIFFSKY_PNAMES = []
ALL_DIFFSKY_PNAMES.extend(MAH_PNAMES)
ALL_DIFFSKY_PNAMES.extend(MS_PNAMES)
ALL_DIFFSKY_PNAMES.extend(Q_PNAMES)
ALL_DIFFSKY_PNAMES.extend(FBULGE_PNAMES)
ALL_DIFFSKY_PNAMES.extend(BURSTSHAPE_PNAMES)
ALL_DIFFSKY_PNAMES.extend(["fknot", "fburst", "redshift"])


class DiffskyParams(typing.NamedTuple):
    """NamedTuple storing parameters of a Diffsky galaxy"""

    mah_params: np.float32
    ms_params: np.float32
    q_params: np.float32
    fburst: np.float32
    burstshape_params: np.float32
    fbulge_params: np.float32
    fknot: np.float32


def get_healpixel_bname_from_ra_dec_z(
    ra, dec, z, nside=ROMAN_RUBIN_2023_NSIDE, hpix_pat=ROMAN_RUBIN_2023_HPIX_PAT
):
    """Retrieve the healpixel filename that contains the input sky coordinates.

    Parameters
    ----------
    ra : float
        ra coordinate of the galaxy of interest

    dec : float
        dec coordinate of the galaxy of interest

    z : float
        redshift of the galaxy of interest

    nside : int, optional
        Number of healpixel divisions.
        Default is defaults.ROMAN_RUBIN_2023_NSIDE

    hpix_pat : string, optional
        Filename pattern of the roman_rubin_2023 mock.
        Default is defaults.ROMAN_RUBIN_2023_HPIX_PAT

    Returns
    -------
    basename : string
        Basename of the healpixel file storing the galaxy of interest

    """
    if z < 1:
        zlo, zhi = 0, 1
    elif 1 <= z < 2:
        zlo, zhi = 1, 2
    elif 2 <= z < 3:
        zlo, zhi = 2, 3
    else:
        raise ValueError("Input redshift z={0} must be in the range 0 < z < 3")

    pix = hp.ang2pix(nside, ra, dec, lonlat=True)
    bname = hpix_pat.format(zlo, zhi, pix)
    return bname


def load_healpixel(fn, patlist=None):
    """Load a Diffsky healpixel from hdf5, concatenating data stored by snapshot

    Parameters
    ----------
    fn : string
        Path to the hdf5 file storing the healpixel

    patlist : list of strings, optional
        List of column name patterns used to retrieve extra columns from the healpixel.
        For example, to select all columns related to LSST photometry,
        setting patlist=('LSST', ) will retrieve all columns in which
        'LSST' appears somewhere in the column name.
        Default behavior is to return all available columns.
        Note that all the Diffsky model parameters are always returned.

    Returns
    -------
    data : OrderedDict

    metadata : OrderedDict

    Notes
    -----
    This standalone function can be used to load diffsky data from disk.
    Each healpixel of lsstdesc-diffsky data is stored on disk such that galaxies
    at different simulation snapshots are partitioned into separate hdf5 datasets.
    The load_diffsky_healpixel function concatenates all these separate datasets
    into flat ndarrays.

    DESC members working at NERSC can instead use the GCR:
    https://github.com/yymao/generic-catalog-reader

    For example usage of the GCR with diffsky,
    see lsstdesc-diffsky/notebooks/demo_load_catalog.ipynb

    """
    data_collection, metadata = collect_healpixel_data(fn, patlist=patlist)
    data = _flatten_data_collection(data_collection)
    return data, metadata


def _get_extra_colnames(all_keys, patlist):
    if patlist is None:
        return all_keys
    else:
        extra_colnames = []
        for pat in patlist:
            extra_colnames.extend([key for key in all_keys if pat in key])
        return extra_colnames


def load_diffsky_params(cat):
    """Load the collection of parameters that determine the SEDs of Diffsky galaxies.
    Results are returned as ndarrays of the shapes expected by the convenience functions
    used to compute SEDs and photometry of diffsky galaxies.

    Parameters
    ----------
    fn : string
        Path to the hdf5 file storing the healpixel

    Returns
    -------
    DiffskyParams : NamedTuple with the following entries

        mah_params : ndarray, shape (ngals, 4)
            Diffmah params specifying the mass assembly of the dark matter halo
            diffmah_params = (logm0, logtc, early_index, late_index)

        ms_params : ndarray, shape (ngals, 5)
            Diffstar params for the star-formation effiency
            and gas consumption timescale
            ms_params = (lgmcrit, lgy_at_mcrit, indx_lo, indx_hi, tau_dep)

        q_params : ndarray, shape (ngals, 4)
            Diffstar quenching params, (lg_qt, qlglgdt, lg_drop, lg_rejuv)

        fburst : ndarray, shape (ngals, )
            Fraction of stellar mass formed in a recent burst

        burstshape_params : ndarray, shape (ngals, 2)
            Parameters controlling the distribution of stellar ages in the recent burst

        fbulge_params : ndarray, shape (ngals, 2)
            Parameters controlling the disk/bulge decomposition

        fknot : ndarray, shape (ngals, )
            Fraction of disk mass located in bursty star-forming knots

    """
    mah_params = np.vstack([cat[key] for key in MAH_PNAMES]).T
    ms_params = np.vstack([cat[key] for key in MS_PNAMES]).T
    q_params = np.vstack([cat[key] for key in Q_PNAMES]).T
    fburst = np.array(cat["fburst"])
    burstshape_params = np.vstack([cat[key] for key in BURSTSHAPE_PNAMES]).T
    fbulge_params = np.vstack([cat[key] for key in FBULGE_PNAMES]).T
    fknot = np.array(cat["fknot"])
    return DiffskyParams(
        mah_params, ms_params, q_params, fburst, burstshape_params, fbulge_params, fknot
    )


def collect_healpixel_data(fn, patlist):
    with h5py.File(fn, "r") as hdf:
        metadataset = hdf["metaData"]
        metadata = OrderedDict()
        for key in metadataset.keys():
            metadata[key] = metadataset[key][...]

        _snaplist = [key for key in hdf.keys() if key != "metaData"]
        snapnums = sorted([int(key) for key in _snaplist])[::-1]
        for snapnum in snapnums:
            dataset = hdf[str(snapnum)]
            if len(dataset) > 0:
                all_keys = list(dataset.keys())
                break
        extra_colnames = _get_extra_colnames(all_keys, patlist)
        desired_dataset_colnames = ALL_DIFFSKY_PNAMES.copy()
        desired_dataset_colnames.extend(extra_colnames)

        data_collection = OrderedDict()
        for snapnum in snapnums:
            snapkey = str(snapnum)
            dataset = hdf[snapkey]
            d = OrderedDict()

            if len(dataset) > 0:
                for key in desired_dataset_colnames:
                    d[key] = dataset[key][...]

                data_collection[snapkey] = d

    return data_collection, metadata


def _flatten_data_collection(data_collection):
    snapkeys = list(data_collection.keys())
    data_colnames = list(data_collection[snapkeys[0]].keys())
    exkey = data_colnames[0]
    ngals = [c[exkey].shape[0] for c in data_collection.values()]
    ngal_tot = sum(ngals)

    scalar_dtypes = [data_collection[snapkeys[0]][key].dtype for key in data_colnames]
    scalar_ndarrays = [np.zeros(ngal_tot, dtype=dt) for dt in scalar_dtypes]
    snapshot_ndarray = np.zeros(ngal_tot, dtype=np.int64)

    ifirst = 0
    for i, (snapkey, dataset) in enumerate(data_collection.items()):
        ngals_snap = ngals[i]
        ilast = ifirst + ngals_snap

        snapshot_ndarray[ifirst:ilast] = int(snapkey)
        for ndarray, key in zip(scalar_ndarrays, data_colnames):
            ndarray[ifirst:ilast] = dataset[key]

        ifirst = ilast

    mockout = OrderedDict()
    mockout["snapnum"] = snapshot_ndarray
    for ndarray, key in zip(scalar_ndarrays, data_colnames):
        mockout[key] = ndarray

    return mockout
