import numpy as np
from diffstar.defaults import SFR_MIN
from dsps.utils import _jax_get_dt_array
from jax import numpy as jnp

from ..constants import MAH_PNAMES, MS_PNAMES, Q_PNAMES
from . import photometry_interpolation_kernels as pik


def get_diff_params(
    data,
    mah_keys=MAH_PNAMES,
    ms_keys=MS_PNAMES,
    q_keys=Q_PNAMES,
):
    print(".....Retrieving mah, ms and q params")

    dlen = len(data)
    mah_params = np.zeros((dlen, len(mah_keys)))
    ms_params = np.zeros((dlen, len(ms_keys)))
    q_params = np.zeros((dlen, len(q_keys)))
    for keys, array in zip(
        [mah_keys, ms_keys, q_keys], [mah_params, ms_params, q_params]
    ):
        for n, colname in enumerate(keys):
            array[:, n] = data[colname].value

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


def get_logsm_sfr_obs(sfh_table, t_obs, t_table):
    """
    Calculate logsm_obs and sfr_obs from star-formation history

    Parameters
    ----------
    sfh_table : array of shape (n_gals, n_t_table)

    t_obs : array of shape (n_gals, )

    t_table : array of shape (n_t_table, )

    Returns
    -------
    logsm_obs : array of shape (n_gals, )

    sfr_obs : array of shape (n_gals, )

    """

    dt_gyr = _jax_get_dt_array(t_table)

    sfh = jnp.where(sfh_table < SFR_MIN, SFR_MIN, sfh_table)
    smh = jnp.cumsum(sfh * dt_gyr, axis=1) * 1e9
    logsmh = jnp.log10(smh)
    logsm_obs = pik._interp_vmap(t_obs, t_table, logsmh)
    sfr_obs = pik._interp_vmap(t_obs, t_table, sfh_table)

    return logsm_obs, sfr_obs
