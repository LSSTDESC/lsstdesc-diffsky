"""
"""
import numpy as np
import os
import re
import galsim.roman as roman
from jax import numpy as jnp
from dsps.data_loaders import load_transmission_curve
from dsps.data_loaders.defaults import TransmissionCurve

__all__ = (
           "assemble_filter_data",
           "get_filter_wave_trans",
          )


def assemble_filter_data(drn, filters):
    """
    Assemble requested filter data into numpy structured arrays of identical lengths
    """
    filter_dict = {
        "locs": {
            "lsstdsps": "filters/lsst_{}_transmission.h5",
            "lsst": "LSST_Imsim/total_{}.dat",
            "hsc": "HSC/{}_HSC.txt",
            "hscbv": "filters/suprimecam_{}_transmission.h5",
            "sdss": "SDSS/{}_SDSS.res",
            "uvista": "COSMOS/COSMOS_UVISTA_{}.h5",
            "roman": "import",
        },
        "bands": {
            "lsstdsps": ("u", "g", "r", "i", "z", "y"),
            "lsst": ("u", "g", "r", "i", "z", "y"),
            "hsc": ("g", "r", "i", "z", "y"),
            "hscbv": ("b", "g", "r", "i", "v", "z"),
            "sdss": ("u", "g", "r", "i", "z"),
            "uvista": ("Y", "H", "J", "Ks"),
            "roman": ("R062", "Z087", "Y106", "J129", "W146", "H158", "F184", "K213"),
        },
        "conversion_to_AA": {
            "lsstdsps": 1.0,
            "lsst": 10.0,
            "hsc": 1.0,
            "hscbv": 1.0,
            "sdss": 1.0,
            "uvista": 1.0,
            "roman": 10.0,
        },

    }

    # Interpolate the filters so that they all have the same length
    filter_size = 0
    filter_specs = []
    for f in filters:
        fwpat = filter_dict["locs"][f]
        if ".h5" in fwpat:
            filter_spec = [
                load_transmission_curve(os.path.join(drn, fwpat.format(band)))
                for band in filter_dict["bands"][f]
            ]
            if filter_dict["conversion_to_AA"][f] != 1:
                filter_spec = [
                    TransmissionCurve(ff.wave*filter_dict["conversion_to_AA"][f],
                                      ff.transmission)
                    for ff in filter_spec
                ]
        elif "import" in fwpat:
            if 'roman' in f:
                rbps = roman.getBandpasses()
                filter_spec = [TransmissionCurve(
                    rbps[band].wave_list*filter_dict["conversion_to_AA"][f],
                    rbps[band](rbps[band].wave_list))
                    for band in filter_dict["bands"][f]
                ]
        else:
            filter_spec = [
                np.loadtxt(
                    os.path.join(drn, fwpat.format(band)),
                    dtype=[("wave", "<f4"), ("transmission", "<f4")],
                    usecols=(0, 1),
                )
                for band in filter_dict["bands"][f]
            ]
            filter_spec = [
                TransmissionCurve(ff["wave"]*filter_dict["conversion_to_AA"][f],
                                  ff["transmission"]) for ff in filter_spec]

        filter_size = max(filter_size, np.max([f.wave.shape[0] for f in filter_spec]))
        filter_specs.append(filter_spec)

    # interpolate arrays to match filter_size
    filter_waves_out = []
    filter_trans_out = []
    for f in [f for filter_spec in filter_specs for f in filter_spec]:  # flatten list
        wave, trans = f.wave, f.transmission
        xout = np.linspace(wave.min(), wave.max(), filter_size)
        yout = np.interp(xout, wave, trans)
        filter_waves_out.append(xout)
        filter_trans_out.append(yout)

    dt_list = []
    # prepare list of ndarray dtype
    filter_names = []
    for f in filters:
        for b in filter_dict["bands"][f]:
            filter_name = "{}_{}".format(f.upper(), b)
            colname = "{}_filter_wave".format(filter_name)
            dt_list.append((colname, "f4"))
            colname = "{}_filter_trans".format(filter_name)
            dt_list.append((colname, "f4"))
            filter_names.append(filter_name)
    dt = np.dtype(dt_list)

    filter_data = np.zeros(filter_size, dtype=dt)
    for ifilter, filter_name in enumerate(filter_names):
        colname = "{0}_filter_wave".format(filter_name)
        filter_data[colname] = filter_waves_out[ifilter]
        colname = "{0}_filter_trans".format(filter_name)
        filter_data[colname] = filter_trans_out[ifilter]

    return filter_data


def get_filter_wave_trans(filter_data):
    wave_keys = [k for k in filter_data.dtype.names if "wave" in k]
    # print(wave_keys)
    trans_keys = [k for k in filter_data.dtype.names if "trans" in k]
    filter_waves = jnp.array([filter_data[key] for key in wave_keys])
    filter_trans = jnp.array([filter_data[key] for key in trans_keys])
    # print(filter_waves.shape, filter_waves.dtype.names)
    filter_keys = [re.sub("_filter_wave", "", k) for k in wave_keys]

    return filter_waves, filter_trans, filter_keys
