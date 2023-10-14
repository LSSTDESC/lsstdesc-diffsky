"""JAX-based implementation of the dust population model in Nagaraj+22.
See https://arxiv.org/abs/2202.05102 for details."""
from collections import OrderedDict

import numpy as np
from jax import jit as jjit
from jax import numpy as jnp
from jax import random as jran

from ..dspspop.nagaraj22_dust import (
    DELTA_PDICT,
    TAU_BOUNDS_PDICT,
    TAU_PDICT,
    _get_median_dust_params_kern,
)


def mc_generate_dust_params(ran_key, logsm, logssfr, redshift, **kwargs):
    """Generate dust_params array that should be passed to precompute_dust_attenuation

    Parameters
    ----------
    ran_key : JAX random seed
        Instance of jax.random.PRNGKey(seed), where seed is any integer

    logsm : float or ndarray of shape (n_gals, )
        Base-10 log of stellar mass in units of Msun assuming h=0.7

    logssfr : float or ndarray of shape (n_gals, )
        Base-10 log of SFR/Mstar in units of yr^-1

    redshift : float or ndarray of shape (n_gals, )

    Returns
    -------
    dust_params : ndarray of shape (n_gals, 3)

    """
    tau_pdict = TAU_PDICT.copy()
    tau_pdict.update((k, kwargs[k]) for k in tau_pdict.keys() & kwargs.keys())
    tau_params = np.array(list(tau_pdict.values()))

    delta_pdict = DELTA_PDICT.copy()
    delta_pdict.update((k, kwargs[k]) for k in delta_pdict.keys() & kwargs.keys())
    delta_params = np.array(list(delta_pdict.values()))

    logsm, logssfr, redshift = get_1d_arrays(logsm, logssfr, redshift)

    dust_params = mc_generate_dust_params_kern(
        ran_key, logsm, logssfr, redshift, tau_params, delta_params
    )
    return dust_params


@jjit
def mc_generate_dust_params_kern(
    ran_key, logsm, logssfr, redshift, tau_params, delta_params
):
    delta_key, av_key = jran.split(ran_key, 2)
    n = logsm.size

    median_eb, median_delta, median_av = _get_median_dust_params_kern(
        logsm, logssfr, redshift, tau_params, delta_params
    )
    delta_lgav = jran.uniform(av_key, minval=-0.2, maxval=0.2, shape=(n,))
    lgav = delta_lgav + jnp.log10(median_av)
    av = 10**lgav

    delta = median_delta + jran.uniform(delta_key, minval=-0.1, maxval=0.1, shape=(n,))
    eb = median_eb + jran.uniform(delta_key, minval=-0.15, maxval=0.15, shape=(n,))

    dust_params = jnp.array((eb, delta, av)).T

    return dust_params


def mc_generate_alt_dustpop_params(ran_key):
    """Generate a dictionary of alternative dustpop parameters

    Parameters
    ----------
    ran_key : JAX random seed
        Instance of jax.random.PRNGKey(seed), where seed is any integer

    Returns
    -------
    pdict : OrderedDict
        Dictionary of dustpop parameters with different values than
        those appearing in nagaraj22_dust.TAU_PDICT

    Notes
    -----
    The returned pdict can be passed as the tau_params arguments to the
    dustpop.mc_generate_dust_params function

    """
    pdict = OrderedDict()
    for pname, bounds in TAU_BOUNDS_PDICT.items():
        bounds = TAU_BOUNDS_PDICT[pname]
        pkey, ran_key = jran.split(ran_key, 2)
        u = jran.uniform(pkey, minval=0, maxval=1, shape=(1,))
        alt_val = bounds[0] + u * (bounds[1] - bounds[0])
        pdict[pname] = float(alt_val)
    return pdict


def get_1d_arrays(*args, jax_arrays=False):
    """Return a list of ndarrays of the same length.

    Each arg must be either an ndarray of shape (npts, ), or a scalar.

    """
    results = [jnp.atleast_1d(arg) for arg in args]
    sizes = [arr.size for arr in results]
    npts = max(sizes)
    msg = "All input arguments should be either a float or ndarray of shape ({0}, )"
    assert set(sizes) <= set((1, npts)), msg.format(npts)

    if jax_arrays:
        result = [jnp.zeros(npts).astype(arr.dtype) + arr for arr in results]
    else:
        result = [np.zeros(npts).astype(arr.dtype) + arr for arr in results]
    return result
