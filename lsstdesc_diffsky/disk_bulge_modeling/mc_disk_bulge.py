"""
"""
import numpy as np
from diffstar.fitting_helpers.stars import _integrate_sfr
from dsps.constants import SFR_MIN
from dsps.utils import _jax_get_dt_array
from jax import jit as jjit
from jax import random as jran
from jax import vmap

from .disk_bulge_kernels import (
    _bulge_sfh,
    _get_params_from_u_params_vmap,
    calc_tform_pop,
)

_A = (None, 0, 0)
_bulge_sfh_vmap = jjit(vmap(_bulge_sfh, in_axes=_A))

_B = (0, None)
_integrate_sfr_vmap = jjit(vmap(_integrate_sfr, in_axes=_B))


def mc_disk_bulge(ran_key, tarr, sfh_pop):
    """Decompose input SFHs into disk and bulge contributions

    Parameters
    ----------
    ran_key : jax.random.PRNGKey

    tarr : ndarray, shape (n_t, )

    sfh_pop : ndarray, shape (n_gals, n_t)

    Returns
    -------
    smh : ndarray, shape (n_gals, n_t)
        Stellar mass history of galaxy in units of Msun

    eff_bulge : ndarray, shape (n_gals, n_t)
        History of in-situ bulge growth efficiency for every galaxy

    sfh_bulge : ndarray, shape (n_gals, n_t)
        Star formation history of bulge in units of Msun/yr

    smh_bulge : ndarray, shape (n_gals, n_t)
        Stellar mass history of bulge in units of Msun

    bth : ndarray, shape (n_gals, n_t)
        History of bulge-to-total mass ratio of every galaxy

    """
    dtarr = _jax_get_dt_array(tarr)
    sfh_pop = np.where(sfh_pop < SFR_MIN, SFR_MIN, sfh_pop)
    smh_pop = _integrate_sfr_vmap(sfh_pop, dtarr)
    t10 = calc_tform_pop(tarr, smh_pop, 0.1)
    t90 = calc_tform_pop(tarr, smh_pop, 0.9)
    logsm0 = smh_pop[:, -1]

    fbulge_params = generate_frac_bulge_params(ran_key, t10, t90, logsm0)
    tcrit, fbulge_early, fbulge_late = fbulge_params
    params_pop = np.array((tcrit, fbulge_early, fbulge_late)).T
    _res = _bulge_sfh_vmap(tarr, sfh_pop, params_pop)
    smh, eff_bulge, sfh_bulge, smh_bulge, bth = _res
    return smh, eff_bulge, sfh_bulge, smh_bulge, bth


def generate_frac_bulge_params(
    ran_key,
    t10,
    t90,
    logsm0,
    mu_u_tcrit=2,
    delta_mu_u_tcrit=3,
    mu_u_early=20,
    delta_mu_u_early=0.1,
    mu_u_late=20,
    delta_mu_u_late=3,
):
    n = t10.size
    tcrit_key, early_key, late_key = jran.split(ran_key, 3)
    scale_u_tcrit = 10
    u_tcrit_table = [
        mu_u_tcrit - delta_mu_u_tcrit * scale_u_tcrit,
        mu_u_tcrit + delta_mu_u_tcrit * scale_u_tcrit,
    ]
    logsm_table = 8, 11.5
    mu_u_tcrit_pop = np.interp(logsm0, logsm_table, u_tcrit_table)
    mc_u_tcrit = jran.normal(tcrit_key, shape=(n,)) * scale_u_tcrit + mu_u_tcrit_pop

    scale_u_early = 5
    u_early_table = [
        mu_u_early - delta_mu_u_early * scale_u_early,
        mu_u_early + delta_mu_u_early * scale_u_early,
    ]
    mu_u_early_pop = np.interp(logsm0, logsm_table, u_early_table)
    mc_u_early = jran.normal(early_key, shape=(n,)) * scale_u_early + mu_u_early_pop

    scale_u_late = 8
    u_late_table = [
        mu_u_late + delta_mu_u_late * scale_u_late,
        mu_u_late - delta_mu_u_late * scale_u_late,
    ]
    mu_u_late_pop = np.interp(logsm0, logsm_table, u_late_table)
    mc_u_late = jran.normal(late_key, shape=(n,)) * scale_u_late + mu_u_late_pop

    u_params = np.array((mc_u_tcrit, mc_u_early, mc_u_late)).T
    tcrit, fbulge_early, fbulge_late = _get_params_from_u_params_vmap(
        u_params, t10, t90
    )
    return tcrit, fbulge_early, fbulge_late
