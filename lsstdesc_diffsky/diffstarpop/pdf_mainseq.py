"""Model of a main sequence galaxy population calibrated to SMDPL halos."""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

TODAY = 13.8
LGT0 = jnp.log10(TODAY)

LGM_X0, LGM_K = 13.0, 0.5

DEFAULT_SFH_PDF_MAINSEQ_PARAMS = OrderedDict(
    mean_ulgm_mainseq_ylo=10.04,
    mean_ulgm_mainseq_yhi=14.98,
    mean_ulgy_mainseq_ylo=-2.69,
    mean_ulgy_mainseq_yhi=3.83,
    mean_ul_mainseq_ylo=-23.74,
    mean_ul_mainseq_yhi=33.59,
    mean_utau_mainseq_ylo=37.79,
    mean_utau_mainseq_yhi=-34.69,
    cov_ulgm_ulgm_mainseq_ylo=-1.00,
    cov_ulgm_ulgm_mainseq_yhi=-1.00,
    cov_ulgy_ulgy_mainseq_ylo=-1.00,
    cov_ulgy_ulgy_mainseq_yhi=-1.00,
    cov_ul_ul_mainseq_ylo=-1.00,
    cov_ul_ul_mainseq_yhi=-1.00,
    cov_utau_utau_mainseq_ylo=-1.00,
    cov_utau_utau_mainseq_yhi=-1.00,
    cov_ulgy_ulgm_mainseq_ylo=0.00,
    cov_ulgy_ulgm_mainseq_yhi=0.00,
    cov_ul_ulgm_mainseq_ylo=0.00,
    cov_ul_ulgm_mainseq_yhi=0.00,
    cov_ul_ulgy_mainseq_ylo=0.00,
    cov_ul_ulgy_mainseq_yhi=0.00,
    cov_utau_ulgm_mainseq_ylo=0.00,
    cov_utau_ulgm_mainseq_yhi=0.00,
    cov_utau_ulgy_mainseq_ylo=0.00,
    cov_utau_ulgy_mainseq_yhi=0.00,
    cov_utau_ul_mainseq_ylo=0.00,
    cov_utau_ul_mainseq_yhi=0.00,
)


@jjit
def _sigmoid(x, logtc, k, ymin, ymax):
    height_diff = ymax - ymin
    return ymin + height_diff / (1.0 + jnp.exp(-k * (x - logtc)))


@jjit
def _inverse_sigmoid(y, x0=0, k=1, ymin=-1, ymax=1):
    lnarg = (ymax - ymin) / (y - ymin) - 1
    return x0 - jnp.log(lnarg) / k


@jjit
def _fun(x, ymin, ymax):
    return _sigmoid(x, LGM_X0, LGM_K, ymin, ymax)


@jjit
def _fun_cov_diag(x, ymin, ymax):
    _res = 10 ** _sigmoid(x, 13.0, 0.5, ymin, ymax)
    return _res


@jjit
def _bound_cov_diag(x):
    return _sigmoid(x, 0.5, 4.0, 0.0, 1.0)


@jjit
def _bound_cov_offdiag(x):
    return _sigmoid(x, 0.0, 4.0, -1.0, 1.0)


@jjit
def _unbound_cov_diag(x):
    return _inverse_sigmoid(x, 0.5, 4.0, 0.0, 1.0)


@jjit
def _unbound_cov_offdiag(x):
    return _inverse_sigmoid(x, 0.0, 4.0, -1.0, 1.0)


@jjit
def _get_cov_scalar(
    ulgm_ulgm,
    ulgy_ulgy,
    ul_ul,
    utau_utau,
    ulgy_ulgm,
    ul_ulgm,
    ul_ulgy,
    utau_ulgm,
    utau_ulgy,
    utau_ul,
):
    cov = jnp.zeros((4, 4)).astype("f4")
    cov = cov.at[(0, 0)].set(ulgm_ulgm ** 2)
    cov = cov.at[(1, 1)].set(ulgy_ulgy ** 2)
    cov = cov.at[(2, 2)].set(ul_ul ** 2)
    cov = cov.at[(3, 3)].set(utau_utau ** 2)

    cov = cov.at[(1, 0)].set(ulgy_ulgm * ulgy_ulgy * ulgm_ulgm)
    cov = cov.at[(0, 1)].set(ulgy_ulgm * ulgy_ulgy * ulgm_ulgm)
    cov = cov.at[(2, 0)].set(ul_ulgm * ul_ul * ulgm_ulgm)
    cov = cov.at[(0, 2)].set(ul_ulgm * ul_ul * ulgm_ulgm)
    cov = cov.at[(2, 1)].set(ul_ulgy * ul_ul * ulgy_ulgy)
    cov = cov.at[(1, 2)].set(ul_ulgy * ul_ul * ulgy_ulgy)
    cov = cov.at[(3, 0)].set(utau_ulgm * utau_utau * ulgm_ulgm)
    cov = cov.at[(0, 3)].set(utau_ulgm * utau_utau * ulgm_ulgm)
    cov = cov.at[(3, 1)].set(utau_ulgy * utau_utau * ulgy_ulgy)
    cov = cov.at[(1, 3)].set(utau_ulgy * utau_utau * ulgy_ulgy)
    cov = cov.at[(3, 2)].set(utau_ul * utau_utau * ul_ul)
    cov = cov.at[(2, 3)].set(utau_ul * utau_utau * ul_ul)
    return cov


_get_cov_vmap = jjit(vmap(_get_cov_scalar, in_axes=(*[0] * 10,)))


@jjit
def mean_ulgm_mainseq_vs_lgm0(
    lgm0,
    mean_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ulgm_mainseq_ylo"],
    mean_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ulgm_mainseq_yhi"],
):
    return _fun(lgm0, mean_ulgm_mainseq_ylo, mean_ulgm_mainseq_yhi)


@jjit
def mean_ulgy_mainseq_vs_lgm0(
    lgm0,
    mean_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ulgy_mainseq_ylo"],
    mean_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ulgy_mainseq_yhi"],
):
    return _fun(lgm0, mean_ulgy_mainseq_ylo, mean_ulgy_mainseq_yhi)


@jjit
def mean_ul_mainseq_vs_lgm0(
    lgm0,
    mean_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ul_mainseq_ylo"],
    mean_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ul_mainseq_yhi"],
):
    return _fun(lgm0, mean_ul_mainseq_ylo, mean_ul_mainseq_yhi)


@jjit
def mean_utau_mainseq_vs_lgm0(
    lgm0,
    mean_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_utau_mainseq_ylo"],
    mean_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_utau_mainseq_yhi"],
):
    return _fun(lgm0, mean_utau_mainseq_ylo, mean_utau_mainseq_yhi)


@jjit
def cov_ulgm_ulgm_mainseq_vs_lgm0(
    lgm0,
    cov_ulgm_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgm_ulgm_mainseq_ylo"
    ],
    cov_ulgm_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgm_ulgm_mainseq_yhi"
    ],
):
    _res = _fun_cov_diag(lgm0, cov_ulgm_ulgm_mainseq_ylo, cov_ulgm_ulgm_mainseq_yhi)
    return _res


@jjit
def cov_ulgy_ulgy_mainseq_vs_lgm0(
    lgm0,
    cov_ulgy_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgy_mainseq_ylo"
    ],
    cov_ulgy_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgy_mainseq_yhi"
    ],
):
    _res = _fun_cov_diag(lgm0, cov_ulgy_ulgy_mainseq_ylo, cov_ulgy_ulgy_mainseq_yhi)
    return _res


@jjit
def cov_ul_ul_mainseq_vs_lgm0(
    lgm0,
    cov_ul_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ul_mainseq_ylo"],
    cov_ul_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ul_mainseq_yhi"],
):
    _res = _fun_cov_diag(lgm0, cov_ul_ul_mainseq_ylo, cov_ul_ul_mainseq_yhi)
    return _res


@jjit
def cov_utau_utau_mainseq_vs_lgm0(
    lgm0,
    cov_utau_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_utau_mainseq_ylo"
    ],
    cov_utau_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_utau_mainseq_yhi"
    ],
):
    _res = _fun_cov_diag(lgm0, cov_utau_utau_mainseq_ylo, cov_utau_utau_mainseq_yhi)
    return _res


@jjit
def cov_ulgy_ulgm_mainseq_vs_lgm0(
    lgm0,
    cov_ulgy_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgm_mainseq_ylo"
    ],
    cov_ulgy_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgm_mainseq_yhi"
    ],
):
    _res = _fun(lgm0, cov_ulgy_ulgm_mainseq_ylo, cov_ulgy_ulgm_mainseq_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_ul_ulgm_mainseq_vs_lgm0(
    lgm0,
    cov_ul_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgm_mainseq_ylo"],
    cov_ul_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgm_mainseq_yhi"],
):
    _res = _fun(lgm0, cov_ul_ulgm_mainseq_ylo, cov_ul_ulgm_mainseq_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_ul_ulgy_mainseq_vs_lgm0(
    lgm0,
    cov_ul_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgy_mainseq_ylo"],
    cov_ul_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgy_mainseq_yhi"],
):
    _res = _fun(lgm0, cov_ul_ulgy_mainseq_ylo, cov_ul_ulgy_mainseq_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_utau_ulgm_mainseq_vs_lgm0(
    lgm0,
    cov_utau_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgm_mainseq_ylo"
    ],
    cov_utau_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgm_mainseq_yhi"
    ],
):
    _res = _fun(lgm0, cov_utau_ulgm_mainseq_ylo, cov_utau_ulgm_mainseq_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_utau_ulgy_mainseq_vs_lgm0(
    lgm0,
    cov_utau_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgy_mainseq_ylo"
    ],
    cov_utau_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgy_mainseq_yhi"
    ],
):
    _res = _fun(lgm0, cov_utau_ulgy_mainseq_ylo, cov_utau_ulgy_mainseq_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_utau_ul_mainseq_vs_lgm0(
    lgm0,
    cov_utau_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_utau_ul_mainseq_ylo"],
    cov_utau_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_utau_ul_mainseq_yhi"],
):
    _res = _fun(lgm0, cov_utau_ul_mainseq_ylo, cov_utau_ul_mainseq_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def get_default_params(lgm):
    ulgm_MS = mean_ulgm_mainseq_vs_lgm0(lgm)
    ulgy_MS = mean_ulgy_mainseq_vs_lgm0(lgm)
    ul_MS = mean_ul_mainseq_vs_lgm0(lgm)
    utau_MS = mean_utau_mainseq_vs_lgm0(lgm)
    ulgm_ulgm_MS = cov_ulgm_ulgm_mainseq_vs_lgm0(lgm)
    ulgy_ulgy_MS = cov_ulgy_ulgy_mainseq_vs_lgm0(lgm)
    ul_ul_MS = cov_ul_ul_mainseq_vs_lgm0(lgm)
    utau_utau_MS = cov_utau_utau_mainseq_vs_lgm0(lgm)
    ulgy_ulgm_MS = cov_ulgy_ulgm_mainseq_vs_lgm0(lgm)
    ul_ulgm_MS = cov_ul_ulgm_mainseq_vs_lgm0(lgm)
    ul_ulgy_MS = cov_ul_ulgy_mainseq_vs_lgm0(lgm)
    utau_ulgm_MS = cov_utau_ulgm_mainseq_vs_lgm0(lgm)
    utau_ulgy_MS = cov_utau_ulgy_mainseq_vs_lgm0(lgm)
    utau_ul_MS = cov_utau_ul_mainseq_vs_lgm0(lgm)

    all_params = (
        ulgm_MS,
        ulgy_MS,
        ul_MS,
        utau_MS,
        ulgm_ulgm_MS,
        ulgy_ulgy_MS,
        ul_ul_MS,
        utau_utau_MS,
        ulgy_ulgm_MS,
        ul_ulgm_MS,
        ul_ulgy_MS,
        utau_ulgm_MS,
        utau_ulgy_MS,
        utau_ul_MS,
    )
    return all_params


@jjit
def get_smah_means_and_covs_mainseq(
    logmp_arr,
    mean_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ulgm_mainseq_ylo"],
    mean_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ulgm_mainseq_yhi"],
    mean_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ulgy_mainseq_ylo"],
    mean_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ulgy_mainseq_yhi"],
    mean_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ul_mainseq_ylo"],
    mean_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ul_mainseq_yhi"],
    mean_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_utau_mainseq_ylo"],
    mean_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_utau_mainseq_yhi"],
    cov_ulgm_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgm_ulgm_mainseq_ylo"
    ],
    cov_ulgm_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgm_ulgm_mainseq_yhi"
    ],
    cov_ulgy_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgy_mainseq_ylo"
    ],
    cov_ulgy_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgy_mainseq_yhi"
    ],
    cov_ul_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ul_mainseq_ylo"],
    cov_ul_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ul_mainseq_yhi"],
    cov_utau_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_utau_mainseq_ylo"
    ],
    cov_utau_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_utau_mainseq_yhi"
    ],
    cov_ulgy_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgm_mainseq_ylo"
    ],
    cov_ulgy_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgm_mainseq_yhi"
    ],
    cov_ul_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgm_mainseq_ylo"],
    cov_ul_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgm_mainseq_yhi"],
    cov_ul_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgy_mainseq_ylo"],
    cov_ul_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgy_mainseq_yhi"],
    cov_utau_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgm_mainseq_ylo"
    ],
    cov_utau_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgm_mainseq_yhi"
    ],
    cov_utau_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgy_mainseq_ylo"
    ],
    cov_utau_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgy_mainseq_yhi"
    ],
    cov_utau_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_utau_ul_mainseq_ylo"],
    cov_utau_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_utau_ul_mainseq_yhi"],
):

    _res = _get_mean_smah_params_mainseq(
        logmp_arr,
        mean_ulgm_mainseq_ylo,
        mean_ulgm_mainseq_yhi,
        mean_ulgy_mainseq_ylo,
        mean_ulgy_mainseq_yhi,
        mean_ul_mainseq_ylo,
        mean_ul_mainseq_yhi,
        mean_utau_mainseq_ylo,
        mean_utau_mainseq_yhi,
    )

    means_mainseq = jnp.array(_res).T

    covs_mainseq = _get_covs_mainseq(
        logmp_arr,
        cov_ulgm_ulgm_mainseq_ylo,
        cov_ulgm_ulgm_mainseq_yhi,
        cov_ulgy_ulgy_mainseq_ylo,
        cov_ulgy_ulgy_mainseq_yhi,
        cov_ul_ul_mainseq_ylo,
        cov_ul_ul_mainseq_yhi,
        cov_utau_utau_mainseq_ylo,
        cov_utau_utau_mainseq_yhi,
        cov_ulgy_ulgm_mainseq_ylo,
        cov_ulgy_ulgm_mainseq_yhi,
        cov_ul_ulgm_mainseq_ylo,
        cov_ul_ulgm_mainseq_yhi,
        cov_ul_ulgy_mainseq_ylo,
        cov_ul_ulgy_mainseq_yhi,
        cov_utau_ulgm_mainseq_ylo,
        cov_utau_ulgm_mainseq_yhi,
        cov_utau_ulgy_mainseq_ylo,
        cov_utau_ulgy_mainseq_yhi,
        cov_utau_ul_mainseq_ylo,
        cov_utau_ul_mainseq_yhi,
    )
    return means_mainseq, covs_mainseq


@jjit
def _get_mean_smah_params_mainseq(
    lgm,
    mean_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ulgm_mainseq_ylo"],
    mean_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ulgm_mainseq_yhi"],
    mean_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ulgy_mainseq_ylo"],
    mean_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ulgy_mainseq_yhi"],
    mean_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ul_mainseq_ylo"],
    mean_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_ul_mainseq_yhi"],
    mean_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_utau_mainseq_ylo"],
    mean_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["mean_utau_mainseq_yhi"],
):
    ulgm = mean_ulgm_mainseq_vs_lgm0(lgm, mean_ulgm_mainseq_ylo, mean_ulgm_mainseq_yhi)
    ulgy = mean_ulgy_mainseq_vs_lgm0(lgm, mean_ulgy_mainseq_ylo, mean_ulgy_mainseq_yhi)
    ul = mean_ul_mainseq_vs_lgm0(lgm, mean_ul_mainseq_ylo, mean_ul_mainseq_yhi)
    utau = mean_utau_mainseq_vs_lgm0(lgm, mean_utau_mainseq_ylo, mean_utau_mainseq_yhi)
    return ulgm, ulgy, ul, utau


@jjit
def _get_covs_mainseq(
    lgmp_arr,
    cov_ulgm_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgm_ulgm_mainseq_ylo"
    ],
    cov_ulgm_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgm_ulgm_mainseq_yhi"
    ],
    cov_ulgy_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgy_mainseq_ylo"
    ],
    cov_ulgy_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgy_mainseq_yhi"
    ],
    cov_ul_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ul_mainseq_ylo"],
    cov_ul_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ul_mainseq_yhi"],
    cov_utau_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_utau_mainseq_ylo"
    ],
    cov_utau_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_utau_mainseq_yhi"
    ],
    cov_ulgy_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgm_mainseq_ylo"
    ],
    cov_ulgy_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgm_mainseq_yhi"
    ],
    cov_ul_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgm_mainseq_ylo"],
    cov_ul_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgm_mainseq_yhi"],
    cov_ul_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgy_mainseq_ylo"],
    cov_ul_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgy_mainseq_yhi"],
    cov_utau_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgm_mainseq_ylo"
    ],
    cov_utau_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgm_mainseq_yhi"
    ],
    cov_utau_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgy_mainseq_ylo"
    ],
    cov_utau_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgy_mainseq_yhi"
    ],
    cov_utau_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_utau_ul_mainseq_ylo"],
    cov_utau_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_utau_ul_mainseq_yhi"],
):

    _res = _get_cov_params_mainseq(
        lgmp_arr,
        cov_ulgm_ulgm_mainseq_ylo,
        cov_ulgm_ulgm_mainseq_yhi,
        cov_ulgy_ulgy_mainseq_ylo,
        cov_ulgy_ulgy_mainseq_yhi,
        cov_ul_ul_mainseq_ylo,
        cov_ul_ul_mainseq_yhi,
        cov_utau_utau_mainseq_ylo,
        cov_utau_utau_mainseq_yhi,
        cov_ulgy_ulgm_mainseq_ylo,
        cov_ulgy_ulgm_mainseq_yhi,
        cov_ul_ulgm_mainseq_ylo,
        cov_ul_ulgm_mainseq_yhi,
        cov_ul_ulgy_mainseq_ylo,
        cov_ul_ulgy_mainseq_yhi,
        cov_utau_ulgm_mainseq_ylo,
        cov_utau_ulgm_mainseq_yhi,
        cov_utau_ulgy_mainseq_ylo,
        cov_utau_ulgy_mainseq_yhi,
        cov_utau_ul_mainseq_ylo,
        cov_utau_ul_mainseq_yhi,
    )
    return _get_cov_vmap(*_res)


@jjit
def _get_cov_params_mainseq(
    lgm,
    cov_ulgm_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgm_ulgm_mainseq_ylo"
    ],
    cov_ulgm_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgm_ulgm_mainseq_yhi"
    ],
    cov_ulgy_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgy_mainseq_ylo"
    ],
    cov_ulgy_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgy_mainseq_yhi"
    ],
    cov_ul_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ul_mainseq_ylo"],
    cov_ul_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ul_mainseq_yhi"],
    cov_utau_utau_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_utau_mainseq_ylo"
    ],
    cov_utau_utau_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_utau_mainseq_yhi"
    ],
    cov_ulgy_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgm_mainseq_ylo"
    ],
    cov_ulgy_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_ulgy_ulgm_mainseq_yhi"
    ],
    cov_ul_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgm_mainseq_ylo"],
    cov_ul_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgm_mainseq_yhi"],
    cov_ul_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgy_mainseq_ylo"],
    cov_ul_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_ul_ulgy_mainseq_yhi"],
    cov_utau_ulgm_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgm_mainseq_ylo"
    ],
    cov_utau_ulgm_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgm_mainseq_yhi"
    ],
    cov_utau_ulgy_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgy_mainseq_ylo"
    ],
    cov_utau_ulgy_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS[
        "cov_utau_ulgy_mainseq_yhi"
    ],
    cov_utau_ul_mainseq_ylo=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_utau_ul_mainseq_ylo"],
    cov_utau_ul_mainseq_yhi=DEFAULT_SFH_PDF_MAINSEQ_PARAMS["cov_utau_ul_mainseq_yhi"],
):
    ulgm_ulgm = cov_ulgm_ulgm_mainseq_vs_lgm0(
        lgm, cov_ulgm_ulgm_mainseq_ylo, cov_ulgm_ulgm_mainseq_yhi
    )
    ulgy_ulgy = cov_ulgy_ulgy_mainseq_vs_lgm0(
        lgm, cov_ulgy_ulgy_mainseq_ylo, cov_ulgy_ulgy_mainseq_yhi
    )
    ul_ul = cov_ul_ul_mainseq_vs_lgm0(lgm, cov_ul_ul_mainseq_ylo, cov_ul_ul_mainseq_yhi)
    utau_utau = cov_utau_utau_mainseq_vs_lgm0(
        lgm, cov_utau_utau_mainseq_ylo, cov_utau_utau_mainseq_yhi
    )
    ulgy_ulgm = cov_ulgy_ulgm_mainseq_vs_lgm0(
        lgm, cov_ulgy_ulgm_mainseq_ylo, cov_ulgy_ulgm_mainseq_yhi
    )
    ul_ulgm = cov_ul_ulgm_mainseq_vs_lgm0(
        lgm, cov_ul_ulgm_mainseq_ylo, cov_ul_ulgm_mainseq_yhi
    )
    ul_ulgy = cov_ul_ulgy_mainseq_vs_lgm0(
        lgm, cov_ul_ulgy_mainseq_ylo, cov_ul_ulgy_mainseq_yhi
    )
    utau_ulgm = cov_utau_ulgm_mainseq_vs_lgm0(
        lgm, cov_utau_ulgm_mainseq_ylo, cov_utau_ulgm_mainseq_yhi
    )
    utau_ulgy = cov_utau_ulgy_mainseq_vs_lgm0(
        lgm, cov_utau_ulgy_mainseq_ylo, cov_utau_ulgy_mainseq_yhi
    )
    utau_ul = cov_utau_ul_mainseq_vs_lgm0(
        lgm, cov_utau_ul_mainseq_ylo, cov_utau_ul_mainseq_yhi
    )

    cov_params = (
        ulgm_ulgm,
        ulgy_ulgy,
        ul_ul,
        utau_utau,
        ulgy_ulgm,
        ul_ulgm,
        ul_ulgy,
        utau_ulgm,
        utau_ulgy,
        utau_ul,
    )

    return cov_params
