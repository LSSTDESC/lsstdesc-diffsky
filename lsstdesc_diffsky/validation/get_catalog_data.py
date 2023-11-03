import numpy as np
import h5py
import glob
import os
import pickle
from os.path import expanduser
from astropy.table import Table
input_dir = '/lus/eagle/projects/LastJourney/kovacs/Catalog_5000/OR_5000/'
catdir = 'roman_rubin_2023_v1.0.1'
fname = 'roman_rubin_2023_z_*_cutout_{}.hdf5'


def get_fhlist(input_dir, fname, cutout='*'):

    input_files = sorted(glob.glob(os.path.join(input_dir,
                                                fname.format(cutout))))
    # print(input_files)

    fh_list = [h5py.File(i, 'r') for i in input_files]
    return fh_list


def get_table(fh_list, col_list, verbose=False):

    t = Table()
    for col in col_list:
        array = np.asarray([])
        for fh in fh_list:
            for k in fh.keys():
                if 'meta' in k:
                    continue
                if len(fh[k]) == 0:
                    continue
                if len(fh[k][col].shape) > 1:
                    if len(array) == 0:
                        array = fh[k][col]
                    else:
                        array = np.vstack((array, fh[k][col]))
                else:
                    array = np.concatenate((array, fh[k][col]))
        t[col] = array

    if verbose:
        print('Table colnames: {}, {} galaxies'.format(
            t.colnames, len(t[t.colnames[0]])))
    return t


def get_colnames(fh):
    keys = list(fh.keys())
    return fh[keys[0]].keys()


def check_colnames(fh_list, colnames):
    for fh in fh_list:
        print('Checking {}'.format(fh))
        keys = [k for k in list(fh.keys()) if 'meta' not in k]
        for k in keys:
            if len(fh[k]) == 0:
                continue
            cols = list(fh[k].keys())
            if len(cols) != len(colnames):
                print('Length mismatch for step {}'.format(k))
            extra = [c for c in cols if c not in colnames]
            if extra:
                print('Extra columns: {}'.format(', '.join(extra)))
            missing = [c for c in colnames if c not in cols]
            if missing:
                print('Missing columns: {}'.format(', '.join(missing)))

    print("Done")
    return


def get_catalog(input_dir, cutout, xtra_cols=['redshift', 'target_halo_mass']):
    input_files = sorted(glob.glob(os.path.join(input_dir, fname.format(cutout))))
    fh_list = [h5py.File(i, 'r') for i in input_files]
    colnames = get_colnames(fh_list[0])
    filters = [c for c in colnames if 'LSST' in c or 'SDSS' in c]
    t = get_table(fh_list, filters + xtra_cols)
    # t = get_table(fh_list, colnames)

    t, filters, bandlist = add_colors(t)

    return t, filters, bandlist


def add_colors(mags, minmax=True, exclude=['id'], frames=['rest', 'obs'],
               filters=None):
    # adds colors assuming astropy table
    if filters is None:
        filters = []
        for fr in frames:
            for e in exclude:
                filters = filters + \
                    list(set([col.split('_')[0]
                              for col in mags.colnames if fr in col and
                              '-' not in col and e not in col]))
    print('..Adding colors for filters: ', filters)
    bandlist = []
    for f in filters:
        for fr in frames:
            bands = ([col.split('_')[-1]
                      for col in mags.colnames if f in col and fr in col and
                      '-' not in col])
            # check band order
            bands = reorder(bands, frame=fr)
            bandlist.append(bands)
            for b1, b2 in zip(bands[:-1], bands[1:]):
                f1, f2 = '_'.join([f, fr, b1]), '_'.join([f, fr, b2])
                c = b1 + '-' + b2
                col = '_'.join([f, fr, c])
                mags[col] = mags[f1] - mags[f2]
                if minmax:
                    mask = np.isfinite(mags[col])
                    mag_min = np.min(mags[col][mask])
                    mag_max = np.max(mags[col][mask])
                    print('...{}:{} min/max = {:.3g}/{:.3g}'.format(f, c,
                                                                    mag_min,
                                                                    mag_max))
    return mags, filters, bandlist


def reorder(bands, frame='rest'):
    if 'rest' in frame:
        bands = [b.lower() for b in bands]
    bands = sorted(bands)
    if 'u' in bands and bands.index('u') != 0:
        bands.remove('u')
        bands.insert(0, 'u')
    if 'r' in bands and bands.index('r') > bands.index('i'):
        bands.remove('r')
        bands.insert(bands.index('i'), 'r')
    if 'y' in bands and 'z' in bands and bands.index('y') < bands.index('z'):
        bands.remove('y')
        bands.insert(bands.index('z') + 1, 'y')
    if 'rest' in frame:
        bands = [b.upper() for b in bands]

    return bands


def get_zsnaps(pkldir='cosmology/validsky_new_prod/validsky',
               pklfile='AlphaQ_z2ts.pkl'):
    home = expanduser("~")
    pklfile = os.path.join(home, pkldir, pklfile)
    with open(pklfile, 'rb') as f:
        z2ts = pickle.load(f, encoding='latin1')
    zsnaps = np.asarray(sorted([float(k) for k in z2ts.keys()]))

    return zsnaps
