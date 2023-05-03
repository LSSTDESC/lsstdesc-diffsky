import numpy as np
from diffstar.stars import calculate_histories_batch
from .photometry import photometry_interpolation_kernels as pik
from dsps.utils import _jax_get_dt_array


def get_params(
    um_data,
    tid="source_galaxy_halo_id",
    mah_keys=["t0", "logmp_fit", "mah_logtc", "mah_k", "early_index", "late_index"],
    ms_keys=["lgmcrit", "lgy_at_mcrit", "indx_lo", "indx_hi", "tau_dep"],
    q_keys=["qt", "qs", "q_drop", "q_rejuv"],
):
    print(".....Retrieving mah, ms and q params")

    dlen = len(um_data[tid].value)
    mah_params = np.zeros((dlen, len(mah_keys)))
    ms_params = np.zeros((dlen, len(ms_keys)))
    q_params = np.zeros((dlen, len(q_keys)))
    for keys, array in zip(
        [mah_keys, ms_keys, q_keys], [mah_params, ms_params, q_params]
    ):
        for n, colname in enumerate(keys):
            array[:, n] = um_data[colname].value

    return mah_params, ms_params, q_params


def get_log_safe_ssfr(mstar, sfr, loc=-11.8):  # assumes mstar and sfr are 1-d vectors
    ssfr_random = 10 ** np.random.normal(loc=loc, scale=0.25, size=mstar.size)
    # print(ssfr_random[0:10])
    ssfr = sfr / mstar
    msk_bad = (ssfr <= 0) | ~np.isfinite(ssfr)
    log_safe_ssfr = np.where(msk_bad, ssfr_random, ssfr)
    return np.log10(log_safe_ssfr)


def get_log_ssfr_array(t_bpl, mstar, sfr):
    log_ssfr = np.full(mstar.shape, 0.0)
    # print(log_ssfr.shape)
    for s in np.arange(len(t_bpl)):
        log_ssfr[:, s] = get_log_safe_ssfr(mstar[:, s], sfr[:, s])
        nmask = np.isfinite(log_ssfr[:, s])
        if np.count_nonzero(~nmask) > 0:
            print(s, np.count_nonzero(~nmask))

    return log_ssfr


# Compute SFHs from params
def get_sfh_from_params(
    params,
    t,
    fstar_tdelay,
    mah_key="mah_params",
    ms_key="ms_params",
    q_key="q_params",
    sfh_keys=["mstar", "sfr", "fstar", "dmhdt", "log_mah"],
):
    sfh = {}
    print(".......computing SFHs from diffmah/star params for {} times".format(len(t)))
    print(
        ".......using parameters with shapes {}, {}, {}".format(
            params[mah_key].shape, params[ms_key].shape, params[q_key].shape
        )
    )
    _res = calculate_histories_batch(
        t, params[mah_key], params[ms_key], params[q_key], fstar_tdelay
    )
    for n, s in enumerate(sfh_keys):
        sfh[s] = _res[n]

    sfh["logssfr"] = get_log_ssfr_array(t, sfh["mstar"], sfh["sfr"])

    return sfh


# Compute mstar and sfr from params
def get_logsm_sfr_from_params(lgt_table, lgt0, t_obs, mah_params, ms_params, q_params):

    """Calculate star-formation history from diffmah and diffstar parameters

    Parameters
    ----------
    mah_params : array of shape (4, n_gals)

    ms_params : array of shape (5, n_gals)

    q_params : array of shape (4, n_gals)

    lgt_table : array of shape (n_t_table, )

    t_obs : array of shape (n_gals, )

    lgt0 : float

    Returns
    -------
    gal_sfh : array of shape (n_gals, n_t_table)

    logsm_obs : array of shape (n_gals, )

    sfr_obs : array of shape (n_gals, )

    logsm_table: array of shape (n_gals, n_t_table)
    """

    dt_gyr = _jax_get_dt_array(10**lgt_table)

    gal_dmhdt, gal_log_mah, gal_sfh = pik._calc_galhalo_history_vmap(
        lgt_table, dt_gyr, lgt0, mah_params, ms_params, q_params
    )
    logsm_table = pik._calc_logmstar_formed_vmap(gal_sfh, dt_gyr)
    logsm_obs = pik._interp_vmap(t_obs, lgt_table, logsm_table)
    sfr_obs = pik._interp_vmap(t_obs, lgt_table, gal_sfh)

    return logsm_obs, sfr_obs, gal_sfh
