"""
"""
from collections import OrderedDict

import numpy as np
from diffstar.fitting_helpers.stars import _integrate_sfr
from dsps.constants import SFR_MIN
from dsps.utils import _jax_get_dt_array
from jax import jit as jjit
from jax import lax
from jax import numpy as jnp
from jax import vmap

FBULGE_MIN = 0.05
FBULGE_MAX = 0.95

BOUNDING_K = 0.1

DEFAULT_FBULGE_EARLY = 0.75
DEFAULT_FBULGE_LATE = 0.15


DEFAULT_T10, DEFAULT_T90 = 2.0, 9.0
DEFAULT_FBULGE_PDICT = OrderedDict(tcrit=8.0, fbulge_early=0.5, fbulge_late=0.1)
DEFAULT_FBULGE_PARAMS = np.array(list(DEFAULT_FBULGE_PDICT.values()))


@jjit
def _sigmoid(x, x0, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff * lax.logistic(k * (x - x0))


@jjit
def _inverse_sigmoid(y, x0, k, ymin, ymax):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - lax.log(lnarg) / k


@jjit
def _tw_cuml_kern(x, m, h):
    """Triweight kernel version of an err function."""
    z = (x - m) / h
    val = (
        -5 * z**7 / 69984
        + 7 * z**5 / 2592
        - 35 * z**3 / 864
        + 35 * z / 96
        + 1 / 2
    )
    val = jnp.where(z < -3, 0, val)
    val = jnp.where(z > 3, 1, val)
    return val


@jjit
def _tw_sigmoid(x, x0, tw_h, ymin, ymax):
    height_diff = ymax - ymin
    body = _tw_cuml_kern(x, x0, tw_h)
    return ymin + height_diff * body


@jjit
def _bulge_fraction_kernel(t, thalf, frac_early, frac_late, dt):
    """typical values of 10.0, 0.7, 0.1
    frac_late < frac_late is needed to make bulges redder and so that
    bulge fractions increase with stellar mass
    """
    tw_h = dt / 6.0
    return _tw_sigmoid(t, thalf, tw_h, frac_early, frac_late)


@jjit
def _get_u_params_from_params(params, t10, t90):
    tcrit, fbulge_early, fbulge_late = params

    t50 = (t10 + t90) / 2
    u_tcrit = _inverse_sigmoid(tcrit, t50, BOUNDING_K, t10, t90)

    x0 = (FBULGE_MIN + FBULGE_MAX) / 2
    u_fbulge_early = _inverse_sigmoid(
        fbulge_early, x0, BOUNDING_K, FBULGE_MIN, FBULGE_MAX
    )

    x0_late = (fbulge_early + FBULGE_MIN) / 2
    u_fbulge_late = _inverse_sigmoid(
        fbulge_late, x0_late, BOUNDING_K, fbulge_early, FBULGE_MIN
    )

    u_params = u_tcrit, u_fbulge_early, u_fbulge_late
    return u_params


@jjit
def _get_params_from_u_params(u_params, t10, t90):
    u_tcrit, u_fbulge_early, u_fbulge_late = u_params

    t50 = (t10 + t90) / 2
    tcrit = _sigmoid(u_tcrit, t50, BOUNDING_K, t10, t90)

    x0 = (FBULGE_MIN + FBULGE_MAX) / 2
    fbulge_early = _sigmoid(u_fbulge_early, x0, BOUNDING_K, FBULGE_MIN, FBULGE_MAX)

    x0_late = (fbulge_early + FBULGE_MIN) / 2
    fbulge_late = _sigmoid(u_fbulge_late, x0_late, BOUNDING_K, fbulge_early, FBULGE_MIN)

    params = tcrit, fbulge_early, fbulge_late
    return params


@jjit
def _bulge_fraction_vs_tform_u_params(t, t10, t90, u_params):
    params = _get_params_from_u_params(u_params, t10, t90)
    tcrit, fbulge_early, fbulge_late = params
    dt = t90 - t10
    return _bulge_fraction_kernel(t, tcrit, fbulge_early, fbulge_late, dt)


@jjit
def calc_tform_kern(abscissa, xarr, tform_frac):
    fracarr = xarr / xarr[-1]
    return jnp.interp(tform_frac, fracarr, abscissa)


_calc_tform_pop_kern = jjit(vmap(calc_tform_kern, in_axes=[None, 0, None]))


@jjit
def calc_tform_pop(tarr, smh_pop, tform_frac):
    """Calculate the formation time of a population

    Parameters
    ----------
    tarr : ndarray, shape(nt, )

    smh_pop : ndarray, shape(npop, nt)

    tform_frac : float
        Fraction used in the formation time definition
        tform_frac=0.5 corresponds to the half-mass time, for example

    """
    return _calc_tform_pop_kern(tarr, smh_pop, tform_frac)


@jjit
def _bulge_fraction_vs_tform(t, t10, t90, params):
    tcrit, fbulge_early, fbulge_late = params
    dt = t90 - t10
    fbulge = _bulge_fraction_kernel(t, tcrit, fbulge_early, fbulge_late, dt)
    return fbulge


@jjit
def _bulge_sfh(tarr, sfh, params):
    dtarr = _jax_get_dt_array(tarr)
    sfh = jnp.where(sfh < SFR_MIN, SFR_MIN, sfh)
    smh = _integrate_sfr(sfh, dtarr)
    fracmh = smh / smh[-1]
    t10 = jnp.interp(0.1, fracmh, tarr)
    t90 = jnp.interp(0.9, fracmh, tarr)
    fbulge = _bulge_fraction_vs_tform(tarr, t10, t90, params)
    sfh_bulge = fbulge * sfh
    smh_bulge = _integrate_sfr(sfh_bulge, dtarr)
    bth = smh_bulge / smh
    return smh, fbulge, sfh_bulge, smh_bulge, bth


DEFAULT_FBULGE_U_PARAMS = _get_u_params_from_params(
    DEFAULT_FBULGE_PARAMS, DEFAULT_T10, DEFAULT_T90
)
_A = (0, 0, 0)
_get_params_from_u_params_vmap = jjit(vmap(_get_params_from_u_params, in_axes=_A))
