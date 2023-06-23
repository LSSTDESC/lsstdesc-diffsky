import os
import re
import numpy as np
from astropy.table import Table
cosmos_dir = '/lus/eagle/projects/LastJourney/kovacs/COSMOS2020'
restcolor_dir = '/lus/eagle/projects/LastJourney/kovacs/ValidationData/RestColorData'


def get_data(restcolor_dir=restcolor_dir, cosmos_dir=cosmos_dir):

    aux_data = add_rest_color_data({}, restcolor_dir)
    print(aux_data.keys(), aux_data['BC03_rest'].colnames)

    cosmos2020_uncut = read_COSMOS2020(cosmos_dir)
    cosmos2020, cmask = select_COSMOS2020_data(cosmos2020_uncut)
    aux_data['COSMOS2020'] = cosmos2020[cmask]

    return aux_data


def add_rest_color_data(aux_data, restcolor_dir, frame='rest'):

    for k, fn in zip(['BC03', 'Brown'], ['{}_39_rest_colors.out',
                     '{}_129_rest_colors.out']):
        key = '_'.join([k, frame])
        aux_data[key] = Table()
        rcfile = os.path.join(restcolor_dir, fn.format(k))
        data = np.loadtxt(rcfile, skiprows=1, usecols=(3, 4))
        aux_data[key]['G-R'] = data[:, 0]
        aux_data[key]['R-I'] = data[:, 1]

    return aux_data


def read_COSMOS2020(cosmos_dir, cosmos_fn='COSMOS2020_Farmer_processed_hlin.fits',
                    rename=[('_MAG', ''), ('lp_M', 'lp_M_')]):

    fname = os.path.join(cosmos_dir, cosmos_fn)
    cosmos2020 = Table.read(fname, format='fits', hdu=1)

    # select galaxies
    sel_galaxies = (cosmos2020['lp_type'] == 0)
    cosmos2020_gals = cosmos2020[sel_galaxies]
    print('{} objects'.format(len(cosmos2020_gals)))

    # rename some columns for plotting convenience
    if rename is not None:
        cosmos2020_gals = rename_columns(cosmos2020_gals, rename=rename)

    print(cosmos2020_gals.colnames)

    return cosmos2020_gals


def rename_columns(cosmos2020_gals, rename=[('_MAG', ''), ('lp_M', 'lp_M_')]):
    """
    rename some columns for plotting convenience
    """

    for r in rename:
        cols = [c for c in cosmos2020_gals.colnames if r[0] in c]
        for c in cols:
            cosmos2020_gals.rename_column(c, re.sub(r[0], r[1], c))

    return cosmos2020_gals


def add_cosmos2020_colors(cosmos2020, filter_names, filter_types, bands, minmax=False):

    for fname, ftyp in zip(filter_names, filter_types):
        for b1, b2 in zip(bands[:-1], bands[1:]):
            c = b1 + '-' + b2 if ftyp == 'obs' else b1.upper() + '-' + b2.upper()
            col1 = fname.format(b1) if ftyp == 'obs' else fname.format(b1.upper())
            col2 = fname.format(b2) if ftyp == 'obs' else fname.format(b2.upper())
            if col1 in cosmos2020.colnames and col2 in cosmos2020.colnames:
                colc = fname.format(c)
                cosmos2020[colc] = cosmos2020[col1] - cosmos2020[col2]
            if minmax:
                print('{} min/max = {:.3g}/{:.3g}'.format(c,
                      np.min(cosmos2020[colc]), np.max(cosmos2020[colc])))

    return cosmos2020


def select_COSMOS2020_data(cosmos2020, keyname='COSMOS2020 ({})',
                           selections_max=[('HSC_{}', 26.), ('lp_M_{}', -15.0),
                                           ('HSC_g-r', 2.0), ('HSC_r-i', 1.6),
                                           ('HSC_i-z', 1.2), ('HSC_z-y', 1.0),
                                           ('lp_M_G-R', 1.1), ('lp_M_R-I', 1.2),
                                           ('lp_M_I-Z', 1.0), ('lp_M_Z-Y', 1.2),
                                           ('photoz', 3.0), ('lp_mass_med', 12.5)],
                           selections_min=[('lp_M_{}', -25.0),
                                           ('HSC_g-r', -0.5), ('HSC_r-i', -0.5),
                                           ('HSC_i-z', -0.5), ('HSC_z-y', -1.0),
                                           ('lp_M_G-R', -1.2), ('lp_M_R-I', -1.2),
                                           ('lp_M_I-Z', -0.5), ('lp_M_Z-Y', -0.5),
                                           ('lp_mass_med', 6.0)],
                           filter_names=['HSC_{}', 'lp_M_{}'],
                           filter_types=['obs', 'rest'],
                           bands=['u', 'g', 'r', 'i', 'z', 'y'],
                           ):

    # compute colors
    cosmos2020 = add_cosmos2020_colors(cosmos2020, filter_names, filter_types, bands)

    # make selections
    mask = np.ones(len(cosmos2020), dtype=bool)
    for select in selections_min:
        mask = make_selections(cosmos2020, select, mask, bands, minimum=True)

    for select in selections_max:
        mask = make_selections(cosmos2020, select, mask, bands, minimum=False)

    print('Selecting {}/{} objects after all cuts\n'.format(np.count_nonzero(mask),
                                                            len(mask)))

    return cosmos2020, mask


def make_selections(cosmos2020, select, mask, bands, minimum=True):

    for b in bands:
        key = select[0].format(b) if select[1] > 0 else select[0].format(b.upper())
        if key in cosmos2020.colnames:
            if minimum:
                dmask = (cosmos2020[key] >= select[1])
                msg = '>='
            else:
                dmask = (cosmos2020[key] <= select[1])
                msg = '<='
            dcount = np.count_nonzero(dmask)
            print('Selecting {}/{} objects with {} {} {:.1f}'.format(dcount,
                                                                     len(dmask),
                                                                     key, msg,
                                                                     select[1]))

            mask = mask & dmask

    return mask
