import numpy as np
import time
import re
from astropy.table import Table, vstack
from jax import numpy as jnp
from lsstdesc_diffsky.photometry_interpolation import get_interpolated_photometry


def get_filter_wave_trans(filter_data):
    wave_keys = [k for k in filter_data.dtype.names if "wave" in k]
    # print(wave_keys)
    trans_keys = [k for k in filter_data.dtype.names if "trans" in k]
    filter_waves = jnp.array([filter_data[key] for key in wave_keys])
    filter_trans = jnp.array([filter_data[key] for key in trans_keys])
    # print(filter_waves.shape, filter_waves.dtype.names)
    filter_keys = [re.sub("_filter_wave", "", k) for k in wave_keys]

    return filter_waves, filter_trans, filter_keys


def add_colors(mags, minmax=True):
    # adds colors assuming astropy table
    filters = list(set([col.split("_")[0] for col in mags.colnames if "-" not in col]))
    frames = list(set([col.split("_")[1] for col in mags.colnames if "-" not in col]))
    for f in filters:
        for fr in frames:
            bands = [
                col.split("_")[-1]
                for col in mags.colnames
                if f in col and fr in col and "-" not in col
            ]
            for b1, b2 in zip(bands[:-1], bands[1:]):
                f1, f2 = "_".join([f, fr, b1]), "_".join([f, fr, b2])
                c = b1 + "-" + b2
                col = "_".join([f, fr, c])
                mags[col] = mags[f1] - mags[f2]
                if minmax:
                    mask = np.isfinite(mags[col])
                    print(
                        "{}:{} min/max = {:.3g}/{:.3g}".format(
                            f, c, np.min(mags[col][mask]), np.max(mags[col][mask])
                        )
                    )

    return mags


def get_mag_sed_pars(
    t_obs,
    z_obs,
    SED_params,
    lg_met_mean,
    lg_met_scatt,
    log_sm,
    logsm_table,
    attenuation_factors=None,
    skip_mags=False,
):

    print(
        ".....Evaluating colors for {:.2f} <= t <= {:.2f} Gyr".format(
            np.min(t_obs), np.max(t_obs)
        )
    )

    # setup arguments for computing magnitudes
    mags = Table()
    seds = np.asarray([])

    if not skip_mags:
        args = (
            SED_params["ssp_z_table"],
            SED_params["ssp_restmag_table"],
            SED_params["ssp_obsmag_table"],
            SED_params["lgZsun_bin_mids"],
            SED_params["log_age_gyr"],
            SED_params["lgt_table"],
            z_obs,
            t_obs,
            log_sm,
            logsm_table,
            lg_met_mean,
            lg_met_scatt,
            SED_params["LGT0"],
        )

        start = time.time()
        _res = get_interpolated_photometry(
            *args, attenuation_factors=attenuation_factors
        )
        gal_obsmags, gal_restmags = _res

        # add values to table
        for fr, vals in zip(["rest", "obs"], [gal_restmags, gal_obsmags]):
            for k in SED_params["filter_keys"]:
                filt = k.split("_")[0]
                band = k.split("_")[1]
                band = band.upper() if fr == "rest" else band
                colname = "{}_{}_{}".format(filt, fr, band)
                column = SED_params["filter_keys"].index(k)
                mags[colname] = vals[:, column]

        end = time.time()
        print(
            ".......runtime to compute {} galaxies = {:.2f} seconds".format(
                len(z_obs), end - start
            )
        )

    return mags, seds
