import numpy as np
import time
import re
from astropy.table import Table
from jax import numpy as jnp
from diffsky.experimental.photometry_interpolation import get_interpolated_photometry
from dsps.cosmology import flat_wcdm


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
        SED_params,
        gal_z_obs,
        gal_log_sm,
        gal_sfr_table,
        gal_lg_met_mean,
        gal_lg_met_scatt,
        cosmology, w0, wa,
        dust_trans_factors_obs=1.0,
        dust_trans_factors_rest=1.0,
        skip_mags=False):

    # setup arguments for computing magnitudes
    mags = Table()
    mags_nodust = Table()

    cosmo_params = flat_wcdm.CosmoParams(cosmology.Om0, w0, wa,
                                         cosmology.H0.value/100)
    gal_t_table = SED_params['lgt_table']
    print(
        ".....Evaluating mags & colors for {:.4f} <= z <= {:.4f}".format(
            np.min(gal_z_obs), np.max(gal_z_obs)
        )
    )

    if not skip_mags:
        args = (
            SED_params["ssp_z_table"],
            SED_params["ssp_restmag_table"],
            SED_params["ssp_obsmag_table"],
            SED_params["ssp_lgZsun_bin_mids"],
            SED_params["ssp_log_age_gyr"],
            gal_t_table,
            gal_z_obs,
            gal_log_sm,
            gal_sfr_table,
            gal_lg_met_mean,
            gal_lg_met_scatt,
            cosmo_params,
        )

        start = time.time()
        _res = get_interpolated_photometry(
            *args,
            dust_trans_factors_obs=dust_trans_factors_obs,
            dust_trans_factors_rest=dust_trans_factors_rest,
        )
        gal_obsmags, gal_restmags, gal_obsmags_nodust, gal_restmags_nodust = _res

        # add values to tables
        for table, results in zip([mags, mags_nodust],
                                  [[gal_restmags, gal_obsmags],
                                   [gal_restmags_nodust, gal_obsmags_nodust]]):
            for fr, vals in zip(["rest", "obs"], results):
                for k in SED_params["filter_keys"]:
                    filt = k.split("_")[0]
                    band = k.split("_")[1]
                    band = band.upper() if fr == "rest" else band
                    colname = "{}_{}_{}".format(filt, fr, band)
                    column = SED_params["filter_keys"].index(k)
                    table[colname] = vals[:, column]

        end = time.time()
        print(
            ".......runtime to compute {} galaxies = {:.2f} seconds".format(
                len(gal_z_obs), end - start
            )
        )

    return mags, mags_nodust
