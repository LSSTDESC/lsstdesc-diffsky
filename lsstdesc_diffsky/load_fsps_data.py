import numpy as np
import os


def load_sps_data(sps_drn):

    zlegend = np.load(os.path.join(sps_drn, "zlegend.npy"))
    lgZsun_bin_mids = np.log10(zlegend / zlegend[-3])
    log_age_gyr = np.load(os.path.join(sps_drn, "log_age.npy")) - 9
    ssp_wave = np.load(os.path.join(sps_drn, "ssp_spec_wave.npy"))
    ssp_flux = np.load(os.path.join(sps_drn, "ssp_spec_flux_lines.npy"))

    return ssp_wave, ssp_flux, lgZsun_bin_mids, log_age_gyr


def load_filter_data(drn, filters):

    filter_dict = {
        "locs": {
            "lsst": "LSST/lsst_{}_transmission.npy",
            "hsc": "HSC/{}_HSC.txt",
            "sdss": "SDSS/{}_SDSS.res",
        },
        "bands": {
            "lsst": ("u", "g", "r", "i", "z", "y"),
            "hsc": ("g", "r", "i", "z", "y"),
            "sdss": ("u", "g", "r", "i", "z"),
        },
    }

    # Interpolate the filters so that they all have the same length
    filter_size = 0
    filter_specs = []
    for f in filters:
        fwpat = filter_dict["locs"][f]
        if ".npy" in fwpat:
            filter_spec = [
                np.load(os.path.join(drn, fwpat.format(band)))
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
        filter_size = max(filter_size, np.max([f.shape[0] for f in filter_spec]))
        filter_specs.append(filter_spec)

    # print(filter_size, len(filter_specs))
    filter_waves_out = []
    filter_trans_out = []
    for f in [f for filter_spec in filter_specs for f in filter_spec]:  # flatten list
        wave, trans = f["wave"], f["transmission"]
        xout = np.linspace(wave.min(), wave.max(), filter_size)
        yout = np.interp(xout, wave, trans)
        filter_waves_out.append(xout)
        filter_trans_out.append(yout)

    dt_list = []
    # prepare ndarray
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
