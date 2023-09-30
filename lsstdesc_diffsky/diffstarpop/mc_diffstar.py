"""Monte Carlo generator of Diffstar parameters
"""
from collections import OrderedDict, namedtuple

import numpy as np
from diffmah.monte_carlo_diffmah_hiz import mc_diffmah_params_hiz
from diffstar.defaults import DEFAULT_MS_PDICT, DEFAULT_U_Q_PARAMS, LGT0
from diffstar.fitting_helpers.param_clippers import ms_param_clipper, q_param_clipper
from diffstar.kernels.main_sequence_kernels import _get_bounded_sfr_params_vmap
from diffstar.kernels.quenching_kernels import _get_bounded_q_params_vmap
from diffstar.sfh import _get_unbounded_sfr_params
from jax import random as jran

from .pdf_mainseq import get_smah_means_and_covs_mainseq
from .pdf_quenched import get_smah_means_and_covs_quench

_SFHParams = namedtuple(
    "SFHParams",
    ["mah_params", "msk_is_quenched", "ms_params", "q_params"],
)


DEFAULT_UNBOUND_SFR_PARAMS = _get_unbounded_sfr_params(
    *tuple(DEFAULT_MS_PDICT.values())
)
DEFAULT_UNBOUND_SFR_PARAMS_DICT = OrderedDict(
    zip(DEFAULT_MS_PDICT.keys(), DEFAULT_UNBOUND_SFR_PARAMS)
)
DEFAULT_UNBOUND_Q_PARAMS = np.array(DEFAULT_U_Q_PARAMS)
UH = DEFAULT_UNBOUND_SFR_PARAMS_DICT["indx_hi"]

DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ = DEFAULT_UNBOUND_Q_PARAMS.copy()
DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ[0] = 1.9


def mc_diffstarpop(
    ran_key,
    t_obs,
    logmh=None,
    mah_params=None,
    pdf_parameters_MS={},
    pdf_parameters_Q={},
    lgt0=LGT0,
):
    """Generate Monte Carlo realization of the assembly of a population of halos.

    Parameters
    ----------
    ran_key : jax.random.PRNGKey(seed)
        jax random number key

    t_obs : float
        Age of the universe in Gyr at the time of observation

    logmh : ndarray of shape (n_halos, ), optional
        Base-10 log of halo mass of the halo population at the time of observation
        If None, must pass mah_params

    mah_params : ndarray of shape (n_halos, 4), optional
        Diffmah parameters of the halo population
        If None, must pass logmh, in which case DiffmahPop will generate mah_params

    lgt0 : float, optional
        Base-10 log of the age of the universe in Gyr
        Default value is set in diffstar.constants and is approximately 1.14

    **kwargs : floats
        All parameters of the SFH PDF model are accepted as keyword arguments.
        Default values are set by rockstar_pdf_model.DEFAULT_SFH_PDF_PARAMS

    Returns
    -------
    mah_params : ndarray of shape (n_halos, 4)

    msk_is_quenched : ndarray of shape (n_halos, )
        Boolean array indicating whether the galaxy experienced a quenching event

    ms_params : ndarray of shape (n_halos, 5)
        Diffstar main sequence parameters

    q_params : ndarray of shape (n_halos, 4)
        Diffstar quenching parameters

    """
    mah_key, q_key, ms_key, frac_q_key = jran.split(ran_key, 4)

    diffmah_args_msg = "Must input either mah_params or logmh"
    if mah_params is None:
        assert logmh is not None, diffmah_args_msg
        res = mc_diffmah_params_hiz(ran_key, t_obs, logmh, lgt0=lgt0)
        mah_params = np.array(res).T
    elif mah_params is not None:
        assert logmh is None, diffmah_args_msg
        shape_msg = "mah_params.shape={0} must be (n_halos, 4)"
        assert mah_params.shape[1] == 4, shape_msg.format(mah_params.shape)
        logmh = mah_params[:, 0]

    n_halos = mah_params.shape[0]

    _res = get_smah_means_and_covs_quench(logmh, **pdf_parameters_Q)
    frac_quench, means_quench, covs_quench = _res
    frac_quench = np.array(frac_quench)
    means_quench = np.array(means_quench)
    covs_quench = np.array(covs_quench)

    uran = jran.uniform(frac_q_key, shape=(n_halos,))
    msk_is_quenched = np.array(uran < frac_quench)

    n_halos_Q = msk_is_quenched.sum()
    n_halos_MS = n_halos - n_halos_Q

    sfr_params_quench = np.zeros((n_halos_Q, 5))
    sfr_params_mainseq = np.zeros((n_halos_MS, 5))
    q_params_mainseq = np.zeros((n_halos_MS, 4))

    ms_u_params = np.zeros((n_halos, 5))
    q_u_params = np.zeros((n_halos, 4))

    if n_halos_Q > 0:
        mu = means_quench[msk_is_quenched]
        cov = covs_quench[msk_is_quenched]
        sfh_params_quench = jran.multivariate_normal(
            q_key, mean=mu, cov=cov, shape=(n_halos_Q,)
        )
        sfh_params_quench = np.array(sfh_params_quench)

        sfr_params_quench[:, :3] = sfh_params_quench[:, :3]
        sfr_params_quench[:, 3] = UH
        sfr_params_quench[:, 4] = sfh_params_quench[:, 3]

        ms_u_params[msk_is_quenched] = sfr_params_quench
        q_u_params[msk_is_quenched] = sfh_params_quench[:, 4:8]

    if n_halos_MS > 0:
        _res = get_smah_means_and_covs_mainseq(
            logmh[~msk_is_quenched], **pdf_parameters_MS
        )
        means_mainseq, covs_mainseq = _res

        sfh_params_mainseq = jran.multivariate_normal(
            ms_key, mean=means_mainseq, cov=covs_mainseq, shape=(n_halos_MS,)
        )
        sfh_params_mainseq = np.array(sfh_params_mainseq)

        sfr_params_mainseq[:, :3] = sfh_params_mainseq[:, :3]
        sfr_params_mainseq[:, 3] = UH
        sfr_params_mainseq[:, 4] = sfh_params_mainseq[:, 3]
        q_params_mainseq[:, np.arange(4)] = DEFAULT_UNBOUND_Q_PARAMS_MAIN_SEQ

        ms_u_params[~msk_is_quenched] = sfr_params_mainseq
        q_u_params[~msk_is_quenched] = q_params_mainseq

    ms_params = _get_bounded_sfr_params_vmap(ms_u_params)
    q_params = _get_bounded_q_params_vmap(q_u_params)

    ms_params = ms_param_clipper(ms_params)
    q_params = q_param_clipper(q_params)

    ret = mah_params, msk_is_quenched, ms_params, q_params

    return _SFHParams(*ret)
