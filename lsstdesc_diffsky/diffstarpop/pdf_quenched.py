"""Model of a quenched galaxy population calibrated to SMDPL halos."""
from collections import OrderedDict
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

TODAY = 13.8
LGT0 = jnp.log10(TODAY)

LGM_X0, LGM_K = 13.0, 0.5

DEFAULT_SFH_PDF_QUENCH_PARAMS = OrderedDict(
    frac_quench_x0=12.06,
    frac_quench_k=1.27,
    frac_quench_ylo=-0.81,
    frac_quench_yhi=1.78,
    mean_ulgm_quench_ylo=11.75,
    mean_ulgm_quench_yhi=12.32,
    mean_ulgy_quench_ylo=0.96,
    mean_ulgy_quench_yhi=-0.63,
    mean_ul_quench_ylo=-2.23,
    mean_ul_quench_yhi=2.48,
    mean_utau_quench_ylo=23.20,
    mean_utau_quench_yhi=-21.89,
    mean_uqt_quench_ylo=1.67,
    mean_uqt_quench_yhi=-0.01,
    mean_uqs_quench_ylo=-3.61,
    mean_uqs_quench_yhi=4.13,
    mean_udrop_quench_ylo=0.33,
    mean_udrop_quench_yhi=-5.13,
    mean_urej_quench_ylo=1.24,
    mean_urej_quench_yhi=-3.41,
    cov_ulgm_ulgm_quench_ylo=-1.00,
    cov_ulgm_ulgm_quench_yhi=-1.00,
    cov_ulgy_ulgy_quench_ylo=-1.00,
    cov_ulgy_ulgy_quench_yhi=-1.00,
    cov_ul_ul_quench_ylo=-1.00,
    cov_ul_ul_quench_yhi=-1.00,
    cov_utau_utau_quench_ylo=-1.00,
    cov_utau_utau_quench_yhi=-1.00,
    cov_uqt_uqt_quench_ylo=-1.00,
    cov_uqt_uqt_quench_yhi=-1.00,
    cov_uqs_uqs_quench_ylo=-1.00,
    cov_uqs_uqs_quench_yhi=-1.00,
    cov_udrop_udrop_quench_ylo=-1.00,
    cov_udrop_udrop_quench_yhi=-1.00,
    cov_urej_urej_quench_ylo=-1.00,
    cov_urej_urej_quench_yhi=-1.00,
    cov_ulgy_ulgm_quench_ylo=0.00,
    cov_ulgy_ulgm_quench_yhi=0.00,
    cov_ul_ulgm_quench_ylo=0.00,
    cov_ul_ulgm_quench_yhi=0.00,
    cov_ul_ulgy_quench_ylo=0.00,
    cov_ul_ulgy_quench_yhi=0.00,
    cov_utau_ulgm_quench_ylo=0.00,
    cov_utau_ulgm_quench_yhi=0.00,
    cov_utau_ulgy_quench_ylo=0.00,
    cov_utau_ulgy_quench_yhi=0.00,
    cov_utau_ul_quench_ylo=0.00,
    cov_utau_ul_quench_yhi=0.00,
    cov_uqt_ulgm_quench_ylo=0.00,
    cov_uqt_ulgm_quench_yhi=0.00,
    cov_uqt_ulgy_quench_ylo=0.00,
    cov_uqt_ulgy_quench_yhi=0.00,
    cov_uqt_ul_quench_ylo=0.00,
    cov_uqt_ul_quench_yhi=0.00,
    cov_uqt_utau_quench_ylo=0.00,
    cov_uqt_utau_quench_yhi=0.00,
    cov_uqs_ulgm_quench_ylo=0.00,
    cov_uqs_ulgm_quench_yhi=0.00,
    cov_uqs_ulgy_quench_ylo=0.00,
    cov_uqs_ulgy_quench_yhi=0.00,
    cov_uqs_ul_quench_ylo=0.00,
    cov_uqs_ul_quench_yhi=0.00,
    cov_uqs_utau_quench_ylo=0.00,
    cov_uqs_utau_quench_yhi=0.00,
    cov_uqs_uqt_quench_ylo=0.00,
    cov_uqs_uqt_quench_yhi=0.00,
    cov_udrop_ulgm_quench_ylo=0.00,
    cov_udrop_ulgm_quench_yhi=0.00,
    cov_udrop_ulgy_quench_ylo=0.00,
    cov_udrop_ulgy_quench_yhi=0.00,
    cov_udrop_ul_quench_ylo=0.00,
    cov_udrop_ul_quench_yhi=0.00,
    cov_udrop_utau_quench_ylo=0.00,
    cov_udrop_utau_quench_yhi=0.00,
    cov_udrop_uqt_quench_ylo=0.00,
    cov_udrop_uqt_quench_yhi=0.00,
    cov_udrop_uqs_quench_ylo=0.00,
    cov_udrop_uqs_quench_yhi=0.00,
    cov_urej_ulgm_quench_ylo=0.00,
    cov_urej_ulgm_quench_yhi=0.00,
    cov_urej_ulgy_quench_ylo=0.00,
    cov_urej_ulgy_quench_yhi=0.00,
    cov_urej_ul_quench_ylo=0.00,
    cov_urej_ul_quench_yhi=0.00,
    cov_urej_utau_quench_ylo=0.00,
    cov_urej_utau_quench_yhi=0.00,
    cov_urej_uqt_quench_ylo=0.00,
    cov_urej_uqt_quench_yhi=0.00,
    cov_urej_uqs_quench_ylo=0.00,
    cov_urej_uqs_quench_yhi=0.00,
    cov_urej_udrop_quench_ylo=0.00,
    cov_urej_udrop_quench_yhi=0.00,
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
def _fun_Mcrit(x, ymin, ymax):
    return _sigmoid(x, 12.0, 4.0, ymin, ymax)


@jjit
def _fun_QT(x, ymin, ymax):
    return _sigmoid(x, 13.0, 4.0, ymin, ymax)


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
def _bound_fquench(x):
    return _sigmoid(x, 0.5, 4.0, 0.0, 1.0)


@jjit
def _fun_fquench(x, x0, k, ymin, ymax):
    _res = _sigmoid(x, x0, k, ymin, ymax)
    return _bound_fquench(_res)


@jjit
def _get_cov_scalar(
    ulgm_ulgm,
    ulgy_ulgy,
    ul_ul,
    utau_utau,
    uqt_uqt,
    uqs_uqs,
    udrop_udrop,
    urej_urej,
    ulgy_ulgm,
    ul_ulgm,
    ul_ulgy,
    utau_ulgm,
    utau_ulgy,
    utau_ul,
    uqt_ulgm,
    uqt_ulgy,
    uqt_ul,
    uqt_utau,
    uqs_ulgm,
    uqs_ulgy,
    uqs_ul,
    uqs_utau,
    uqs_uqt,
    udrop_ulgm,
    udrop_ulgy,
    udrop_ul,
    udrop_utau,
    udrop_uqt,
    udrop_uqs,
    urej_ulgm,
    urej_ulgy,
    urej_ul,
    urej_utau,
    urej_uqt,
    urej_uqs,
    urej_udrop,
):
    cov = jnp.zeros((8, 8)).astype("f4")
    cov = cov.at[(0, 0)].set(ulgm_ulgm ** 2)
    cov = cov.at[(1, 1)].set(ulgy_ulgy ** 2)
    cov = cov.at[(2, 2)].set(ul_ul ** 2)
    cov = cov.at[(3, 3)].set(utau_utau ** 2)
    cov = cov.at[(4, 4)].set(uqt_uqt ** 2)
    cov = cov.at[(5, 5)].set(uqs_uqs ** 2)
    cov = cov.at[(6, 6)].set(udrop_udrop ** 2)
    cov = cov.at[(7, 7)].set(urej_urej ** 2)

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
    cov = cov.at[(4, 0)].set(uqt_ulgm * uqt_uqt * ulgm_ulgm)
    cov = cov.at[(0, 4)].set(uqt_ulgm * uqt_uqt * ulgm_ulgm)
    cov = cov.at[(4, 1)].set(uqt_ulgy * uqt_uqt * ulgy_ulgy)
    cov = cov.at[(1, 4)].set(uqt_ulgy * uqt_uqt * ulgy_ulgy)
    cov = cov.at[(4, 2)].set(uqt_ul * uqt_uqt * ul_ul)
    cov = cov.at[(2, 4)].set(uqt_ul * uqt_uqt * ul_ul)
    cov = cov.at[(4, 3)].set(uqt_utau * uqt_uqt * utau_utau)
    cov = cov.at[(3, 4)].set(uqt_utau * uqt_uqt * utau_utau)
    cov = cov.at[(5, 0)].set(uqs_ulgm * uqs_uqs * ulgm_ulgm)
    cov = cov.at[(0, 5)].set(uqs_ulgm * uqs_uqs * ulgm_ulgm)
    cov = cov.at[(5, 1)].set(uqs_ulgy * uqs_uqs * ulgy_ulgy)
    cov = cov.at[(1, 5)].set(uqs_ulgy * uqs_uqs * ulgy_ulgy)
    cov = cov.at[(5, 2)].set(uqs_ul * uqs_uqs * ul_ul)
    cov = cov.at[(2, 5)].set(uqs_ul * uqs_uqs * ul_ul)
    cov = cov.at[(5, 3)].set(uqs_utau * uqs_uqs * utau_utau)
    cov = cov.at[(3, 5)].set(uqs_utau * uqs_uqs * utau_utau)
    cov = cov.at[(5, 4)].set(uqs_uqt * uqs_uqs * uqt_uqt)
    cov = cov.at[(4, 5)].set(uqs_uqt * uqs_uqs * uqt_uqt)
    cov = cov.at[(6, 0)].set(udrop_ulgm * udrop_udrop * ulgm_ulgm)
    cov = cov.at[(0, 6)].set(udrop_ulgm * udrop_udrop * ulgm_ulgm)
    cov = cov.at[(6, 1)].set(udrop_ulgy * udrop_udrop * ulgy_ulgy)
    cov = cov.at[(1, 6)].set(udrop_ulgy * udrop_udrop * ulgy_ulgy)
    cov = cov.at[(6, 2)].set(udrop_ul * udrop_udrop * ul_ul)
    cov = cov.at[(2, 6)].set(udrop_ul * udrop_udrop * ul_ul)
    cov = cov.at[(6, 3)].set(udrop_utau * udrop_udrop * utau_utau)
    cov = cov.at[(3, 6)].set(udrop_utau * udrop_udrop * utau_utau)
    cov = cov.at[(6, 4)].set(udrop_uqt * udrop_udrop * uqt_uqt)
    cov = cov.at[(4, 6)].set(udrop_uqt * udrop_udrop * uqt_uqt)
    cov = cov.at[(6, 5)].set(udrop_uqs * udrop_udrop * uqs_uqs)
    cov = cov.at[(5, 6)].set(udrop_uqs * udrop_udrop * uqs_uqs)
    cov = cov.at[(7, 0)].set(urej_ulgm * urej_urej * ulgm_ulgm)
    cov = cov.at[(0, 7)].set(urej_ulgm * urej_urej * ulgm_ulgm)
    cov = cov.at[(7, 1)].set(urej_ulgy * urej_urej * ulgy_ulgy)
    cov = cov.at[(1, 7)].set(urej_ulgy * urej_urej * ulgy_ulgy)
    cov = cov.at[(7, 2)].set(urej_ul * urej_urej * ul_ul)
    cov = cov.at[(2, 7)].set(urej_ul * urej_urej * ul_ul)
    cov = cov.at[(7, 3)].set(urej_utau * urej_urej * utau_utau)
    cov = cov.at[(3, 7)].set(urej_utau * urej_urej * utau_utau)
    cov = cov.at[(7, 4)].set(urej_uqt * urej_urej * uqt_uqt)
    cov = cov.at[(4, 7)].set(urej_uqt * urej_urej * uqt_uqt)
    cov = cov.at[(7, 5)].set(urej_uqs * urej_urej * uqs_uqs)
    cov = cov.at[(5, 7)].set(urej_uqs * urej_urej * uqs_uqs)
    cov = cov.at[(7, 6)].set(urej_udrop * urej_urej * udrop_udrop)
    cov = cov.at[(6, 7)].set(urej_udrop * urej_urej * udrop_udrop)
    return cov


_get_cov_vmap = jjit(vmap(_get_cov_scalar, in_axes=(*[0] * 36,)))


@jjit
def frac_quench_vs_lgm0(
    lgm0,
    frac_quench_x0=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_x0"],
    frac_quench_k=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_k"],
    frac_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_ylo"],
    frac_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_yhi"],
):
    return _fun_fquench(
        lgm0, frac_quench_x0, frac_quench_k, frac_quench_ylo, frac_quench_yhi
    )


@jjit
def mean_ulgm_quench_vs_lgm0(
    lgm0,
    mean_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgm_quench_ylo"],
    mean_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgm_quench_yhi"],
):
    return _fun_Mcrit(lgm0, mean_ulgm_quench_ylo, mean_ulgm_quench_yhi)


@jjit
def mean_ulgy_quench_vs_lgm0(
    lgm0,
    mean_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgy_quench_ylo"],
    mean_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgy_quench_yhi"],
):
    return _fun(lgm0, mean_ulgy_quench_ylo, mean_ulgy_quench_yhi)


@jjit
def mean_ul_quench_vs_lgm0(
    lgm0,
    mean_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ul_quench_ylo"],
    mean_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ul_quench_yhi"],
):
    return _fun(lgm0, mean_ul_quench_ylo, mean_ul_quench_yhi)


@jjit
def mean_utau_quench_vs_lgm0(
    lgm0,
    mean_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_utau_quench_ylo"],
    mean_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_utau_quench_yhi"],
):
    return _fun(lgm0, mean_utau_quench_ylo, mean_utau_quench_yhi)


@jjit
def mean_uqt_quench_vs_lgm0(
    lgm0,
    mean_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqt_quench_ylo"],
    mean_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqt_quench_yhi"],
):
    return _fun(lgm0, mean_uqt_quench_ylo, mean_uqt_quench_yhi)


@jjit
def mean_uqs_quench_vs_lgm0(
    lgm0,
    mean_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqs_quench_ylo"],
    mean_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqs_quench_yhi"],
):
    return _fun(lgm0, mean_uqs_quench_ylo, mean_uqs_quench_yhi)


@jjit
def mean_udrop_quench_vs_lgm0(
    lgm0,
    mean_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_udrop_quench_ylo"],
    mean_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_udrop_quench_yhi"],
):
    return _fun(lgm0, mean_udrop_quench_ylo, mean_udrop_quench_yhi)


@jjit
def mean_urej_quench_vs_lgm0(
    lgm0,
    mean_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_urej_quench_ylo"],
    mean_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_urej_quench_yhi"],
):
    return _fun(lgm0, mean_urej_quench_ylo, mean_urej_quench_yhi)


@jjit
def cov_ulgm_ulgm_quench_vs_lgm0(
    lgm0,
    cov_ulgm_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgm_ulgm_quench_ylo"],
    cov_ulgm_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgm_ulgm_quench_yhi"],
):
    _res = _fun_cov_diag(lgm0, cov_ulgm_ulgm_quench_ylo, cov_ulgm_ulgm_quench_yhi)
    return _res


@jjit
def cov_ulgy_ulgy_quench_vs_lgm0(
    lgm0,
    cov_ulgy_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgy_quench_ylo"],
    cov_ulgy_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgy_quench_yhi"],
):
    _res = _fun_cov_diag(lgm0, cov_ulgy_ulgy_quench_ylo, cov_ulgy_ulgy_quench_yhi)
    return _res


@jjit
def cov_ul_ul_quench_vs_lgm0(
    lgm0,
    cov_ul_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ul_quench_ylo"],
    cov_ul_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ul_quench_yhi"],
):
    _res = _fun_cov_diag(lgm0, cov_ul_ul_quench_ylo, cov_ul_ul_quench_yhi)
    return _res


@jjit
def cov_utau_utau_quench_vs_lgm0(
    lgm0,
    cov_utau_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_utau_quench_ylo"],
    cov_utau_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_utau_quench_yhi"],
):
    _res = _fun_cov_diag(lgm0, cov_utau_utau_quench_ylo, cov_utau_utau_quench_yhi)
    return _res


@jjit
def cov_uqt_uqt_quench_vs_lgm0(
    lgm0,
    cov_uqt_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_uqt_quench_ylo"],
    cov_uqt_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_uqt_quench_yhi"],
):
    _res = _fun_cov_diag(lgm0, cov_uqt_uqt_quench_ylo, cov_uqt_uqt_quench_yhi)
    return _res


@jjit
def cov_uqs_uqs_quench_vs_lgm0(
    lgm0,
    cov_uqs_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqs_quench_ylo"],
    cov_uqs_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqs_quench_yhi"],
):
    _res = _fun_cov_diag(lgm0, cov_uqs_uqs_quench_ylo, cov_uqs_uqs_quench_yhi)
    return _res


@jjit
def cov_udrop_udrop_quench_vs_lgm0(
    lgm0,
    cov_udrop_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_udrop_quench_ylo"
    ],
    cov_udrop_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_udrop_quench_yhi"
    ],
):
    _res = _fun_cov_diag(lgm0, cov_udrop_udrop_quench_ylo, cov_udrop_udrop_quench_yhi)
    return _res


@jjit
def cov_urej_urej_quench_vs_lgm0(
    lgm0,
    cov_urej_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_urej_quench_ylo"],
    cov_urej_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_urej_quench_yhi"],
):
    _res = _fun_cov_diag(lgm0, cov_urej_urej_quench_ylo, cov_urej_urej_quench_yhi)
    return _res


@jjit
def cov_ulgy_ulgm_quench_vs_lgm0(
    lgm0,
    cov_ulgy_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgm_quench_ylo"],
    cov_ulgy_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgm_quench_yhi"],
):
    _res = _fun(lgm0, cov_ulgy_ulgm_quench_ylo, cov_ulgy_ulgm_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_ul_ulgm_quench_vs_lgm0(
    lgm0,
    cov_ul_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgm_quench_ylo"],
    cov_ul_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgm_quench_yhi"],
):
    _res = _fun(lgm0, cov_ul_ulgm_quench_ylo, cov_ul_ulgm_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_ul_ulgy_quench_vs_lgm0(
    lgm0,
    cov_ul_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgy_quench_ylo"],
    cov_ul_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgy_quench_yhi"],
):
    _res = _fun(lgm0, cov_ul_ulgy_quench_ylo, cov_ul_ulgy_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_utau_ulgm_quench_vs_lgm0(
    lgm0,
    cov_utau_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgm_quench_ylo"],
    cov_utau_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgm_quench_yhi"],
):
    _res = _fun(lgm0, cov_utau_ulgm_quench_ylo, cov_utau_ulgm_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_utau_ulgy_quench_vs_lgm0(
    lgm0,
    cov_utau_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgy_quench_ylo"],
    cov_utau_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgy_quench_yhi"],
):
    _res = _fun(lgm0, cov_utau_ulgy_quench_ylo, cov_utau_ulgy_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_utau_ul_quench_vs_lgm0(
    lgm0,
    cov_utau_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ul_quench_ylo"],
    cov_utau_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ul_quench_yhi"],
):
    _res = _fun(lgm0, cov_utau_ul_quench_ylo, cov_utau_ul_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_uqt_ulgm_quench_vs_lgm0(
    lgm0,
    cov_uqt_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgm_quench_ylo"],
    cov_uqt_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgm_quench_yhi"],
):
    _res = _fun(lgm0, cov_uqt_ulgm_quench_ylo, cov_uqt_ulgm_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_uqt_ulgy_quench_vs_lgm0(
    lgm0,
    cov_uqt_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgy_quench_ylo"],
    cov_uqt_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgy_quench_yhi"],
):
    _res = _fun(lgm0, cov_uqt_ulgy_quench_ylo, cov_uqt_ulgy_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_uqt_ul_quench_vs_lgm0(
    lgm0,
    cov_uqt_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ul_quench_ylo"],
    cov_uqt_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ul_quench_yhi"],
):
    _res = _fun(lgm0, cov_uqt_ul_quench_ylo, cov_uqt_ul_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_uqt_utau_quench_vs_lgm0(
    lgm0,
    cov_uqt_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_utau_quench_ylo"],
    cov_uqt_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_utau_quench_yhi"],
):
    _res = _fun(lgm0, cov_uqt_utau_quench_ylo, cov_uqt_utau_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_uqs_ulgm_quench_vs_lgm0(
    lgm0,
    cov_uqs_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgm_quench_ylo"],
    cov_uqs_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgm_quench_yhi"],
):
    _res = _fun(lgm0, cov_uqs_ulgm_quench_ylo, cov_uqs_ulgm_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_uqs_ulgy_quench_vs_lgm0(
    lgm0,
    cov_uqs_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgy_quench_ylo"],
    cov_uqs_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgy_quench_yhi"],
):
    _res = _fun(lgm0, cov_uqs_ulgy_quench_ylo, cov_uqs_ulgy_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_uqs_ul_quench_vs_lgm0(
    lgm0,
    cov_uqs_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ul_quench_ylo"],
    cov_uqs_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ul_quench_yhi"],
):
    _res = _fun(lgm0, cov_uqs_ul_quench_ylo, cov_uqs_ul_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_uqs_utau_quench_vs_lgm0(
    lgm0,
    cov_uqs_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_utau_quench_ylo"],
    cov_uqs_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_utau_quench_yhi"],
):
    _res = _fun(lgm0, cov_uqs_utau_quench_ylo, cov_uqs_utau_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_uqs_uqt_quench_vs_lgm0(
    lgm0,
    cov_uqs_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqt_quench_ylo"],
    cov_uqs_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqt_quench_yhi"],
):
    _res = _fun(lgm0, cov_uqs_uqt_quench_ylo, cov_uqs_uqt_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_udrop_ulgm_quench_vs_lgm0(
    lgm0,
    cov_udrop_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgm_quench_ylo"
    ],
    cov_udrop_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgm_quench_yhi"
    ],
):
    _res = _fun(lgm0, cov_udrop_ulgm_quench_ylo, cov_udrop_ulgm_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_udrop_ulgy_quench_vs_lgm0(
    lgm0,
    cov_udrop_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgy_quench_ylo"
    ],
    cov_udrop_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgy_quench_yhi"
    ],
):
    _res = _fun(lgm0, cov_udrop_ulgy_quench_ylo, cov_udrop_ulgy_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_udrop_ul_quench_vs_lgm0(
    lgm0,
    cov_udrop_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_ul_quench_ylo"],
    cov_udrop_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_ul_quench_yhi"],
):
    _res = _fun(lgm0, cov_udrop_ul_quench_ylo, cov_udrop_ul_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_udrop_utau_quench_vs_lgm0(
    lgm0,
    cov_udrop_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_utau_quench_ylo"
    ],
    cov_udrop_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_utau_quench_yhi"
    ],
):
    _res = _fun(lgm0, cov_udrop_utau_quench_ylo, cov_udrop_utau_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_udrop_uqt_quench_vs_lgm0(
    lgm0,
    cov_udrop_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqt_quench_ylo"],
    cov_udrop_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqt_quench_yhi"],
):
    _res = _fun(lgm0, cov_udrop_uqt_quench_ylo, cov_udrop_uqt_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_udrop_uqs_quench_vs_lgm0(
    lgm0,
    cov_udrop_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqs_quench_ylo"],
    cov_udrop_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqs_quench_yhi"],
):
    _res = _fun(lgm0, cov_udrop_uqs_quench_ylo, cov_udrop_uqs_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_urej_ulgm_quench_vs_lgm0(
    lgm0,
    cov_urej_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgm_quench_ylo"],
    cov_urej_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgm_quench_yhi"],
):
    _res = _fun(lgm0, cov_urej_ulgm_quench_ylo, cov_urej_ulgm_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_urej_ulgy_quench_vs_lgm0(
    lgm0,
    cov_urej_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgy_quench_ylo"],
    cov_urej_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgy_quench_yhi"],
):
    _res = _fun(lgm0, cov_urej_ulgy_quench_ylo, cov_urej_ulgy_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_urej_ul_quench_vs_lgm0(
    lgm0,
    cov_urej_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ul_quench_ylo"],
    cov_urej_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ul_quench_yhi"],
):
    _res = _fun(lgm0, cov_urej_ul_quench_ylo, cov_urej_ul_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_urej_utau_quench_vs_lgm0(
    lgm0,
    cov_urej_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_utau_quench_ylo"],
    cov_urej_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_utau_quench_yhi"],
):
    _res = _fun(lgm0, cov_urej_utau_quench_ylo, cov_urej_utau_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_urej_uqt_quench_vs_lgm0(
    lgm0,
    cov_urej_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqt_quench_ylo"],
    cov_urej_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqt_quench_yhi"],
):
    _res = _fun(lgm0, cov_urej_uqt_quench_ylo, cov_urej_uqt_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_urej_uqs_quench_vs_lgm0(
    lgm0,
    cov_urej_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqs_quench_ylo"],
    cov_urej_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqs_quench_yhi"],
):
    _res = _fun(lgm0, cov_urej_uqs_quench_ylo, cov_urej_uqs_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def cov_urej_udrop_quench_vs_lgm0(
    lgm0,
    cov_urej_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_urej_udrop_quench_ylo"
    ],
    cov_urej_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_urej_udrop_quench_yhi"
    ],
):
    _res = _fun(lgm0, cov_urej_udrop_quench_ylo, cov_urej_udrop_quench_yhi)
    return _bound_cov_offdiag(_res)


@jjit
def get_default_params(lgm):
    frac_quench = frac_quench_vs_lgm0(lgm)
    ulgm_q = mean_ulgm_quench_vs_lgm0(lgm)
    ulgy_q = mean_ulgy_quench_vs_lgm0(lgm)
    ul_q = mean_ul_quench_vs_lgm0(lgm)
    utau_q = mean_utau_quench_vs_lgm0(lgm)
    uqt_q = mean_uqt_quench_vs_lgm0(lgm)
    uqs_q = mean_uqs_quench_vs_lgm0(lgm)
    udrop_q = mean_udrop_quench_vs_lgm0(lgm)
    urej_q = mean_urej_quench_vs_lgm0(lgm)
    ulgm_ulgm_q = cov_ulgm_ulgm_quench_vs_lgm0(lgm)
    ulgy_ulgy_q = cov_ulgy_ulgy_quench_vs_lgm0(lgm)
    ul_ul_q = cov_ul_ul_quench_vs_lgm0(lgm)
    utau_utau_q = cov_utau_utau_quench_vs_lgm0(lgm)
    uqt_uqt_q = cov_uqt_uqt_quench_vs_lgm0(lgm)
    uqs_uqs_q = cov_uqs_uqs_quench_vs_lgm0(lgm)
    udrop_udrop_q = cov_udrop_udrop_quench_vs_lgm0(lgm)
    urej_urej_q = cov_urej_urej_quench_vs_lgm0(lgm)
    ulgy_ulgm_q = cov_ulgy_ulgm_quench_vs_lgm0(lgm)
    ul_ulgm_q = cov_ul_ulgm_quench_vs_lgm0(lgm)
    ul_ulgy_q = cov_ul_ulgy_quench_vs_lgm0(lgm)
    utau_ulgm_q = cov_utau_ulgm_quench_vs_lgm0(lgm)
    utau_ulgy_q = cov_utau_ulgy_quench_vs_lgm0(lgm)
    utau_ul_q = cov_utau_ul_quench_vs_lgm0(lgm)
    uqt_ulgm_q = cov_uqt_ulgm_quench_vs_lgm0(lgm)
    uqt_ulgy_q = cov_uqt_ulgy_quench_vs_lgm0(lgm)
    uqt_ul_q = cov_uqt_ul_quench_vs_lgm0(lgm)
    uqt_utau_q = cov_uqt_utau_quench_vs_lgm0(lgm)
    uqs_ulgm_q = cov_uqs_ulgm_quench_vs_lgm0(lgm)
    uqs_ulgy_q = cov_uqs_ulgy_quench_vs_lgm0(lgm)
    uqs_ul_q = cov_uqs_ul_quench_vs_lgm0(lgm)
    uqs_utau_q = cov_uqs_utau_quench_vs_lgm0(lgm)
    uqs_uqt_q = cov_uqs_uqt_quench_vs_lgm0(lgm)
    udrop_ulgm_q = cov_udrop_ulgm_quench_vs_lgm0(lgm)
    udrop_ulgy_q = cov_udrop_ulgy_quench_vs_lgm0(lgm)
    udrop_ul_q = cov_udrop_ul_quench_vs_lgm0(lgm)
    udrop_utau_q = cov_udrop_utau_quench_vs_lgm0(lgm)
    udrop_uqt_q = cov_udrop_uqt_quench_vs_lgm0(lgm)
    udrop_uqs_q = cov_udrop_uqs_quench_vs_lgm0(lgm)
    urej_ulgm_q = cov_urej_ulgm_quench_vs_lgm0(lgm)
    urej_ulgy_q = cov_urej_ulgy_quench_vs_lgm0(lgm)
    urej_ul_q = cov_urej_ul_quench_vs_lgm0(lgm)
    urej_utau_q = cov_urej_utau_quench_vs_lgm0(lgm)
    urej_uqt_q = cov_urej_uqt_quench_vs_lgm0(lgm)
    urej_uqs_q = cov_urej_uqs_quench_vs_lgm0(lgm)
    urej_udrop_q = cov_urej_udrop_quench_vs_lgm0(lgm)

    all_params = (
        frac_quench,
        ulgm_q,
        ulgy_q,
        ul_q,
        utau_q,
        uqt_q,
        uqs_q,
        udrop_q,
        urej_q,
        ulgm_ulgm_q,
        ulgy_ulgy_q,
        ul_ul_q,
        utau_utau_q,
        uqt_uqt_q,
        uqs_uqs_q,
        udrop_udrop_q,
        urej_urej_q,
        ulgy_ulgm_q,
        ul_ulgm_q,
        ul_ulgy_q,
        utau_ulgm_q,
        utau_ulgy_q,
        utau_ul_q,
        uqt_ulgm_q,
        uqt_ulgy_q,
        uqt_ul_q,
        uqt_utau_q,
        uqs_ulgm_q,
        uqs_ulgy_q,
        uqs_ul_q,
        uqs_utau_q,
        uqs_uqt_q,
        udrop_ulgm_q,
        udrop_ulgy_q,
        udrop_ul_q,
        udrop_utau_q,
        udrop_uqt_q,
        udrop_uqs_q,
        urej_ulgm_q,
        urej_ulgy_q,
        urej_ul_q,
        urej_utau_q,
        urej_uqt_q,
        urej_uqs_q,
        urej_udrop_q,
    )
    return all_params


@jjit
def get_smah_means_and_covs_quench(
    logmp_arr,
    frac_quench_x0=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_x0"],
    frac_quench_k=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_k"],
    frac_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_ylo"],
    frac_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["frac_quench_yhi"],
    mean_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgm_quench_ylo"],
    mean_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgm_quench_yhi"],
    mean_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgy_quench_ylo"],
    mean_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgy_quench_yhi"],
    mean_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ul_quench_ylo"],
    mean_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ul_quench_yhi"],
    mean_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_utau_quench_ylo"],
    mean_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_utau_quench_yhi"],
    mean_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqt_quench_ylo"],
    mean_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqt_quench_yhi"],
    mean_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqs_quench_ylo"],
    mean_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqs_quench_yhi"],
    mean_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_udrop_quench_ylo"],
    mean_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_udrop_quench_yhi"],
    mean_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_urej_quench_ylo"],
    mean_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_urej_quench_yhi"],
    cov_ulgm_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgm_ulgm_quench_ylo"],
    cov_ulgm_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgm_ulgm_quench_yhi"],
    cov_ulgy_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgy_quench_ylo"],
    cov_ulgy_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgy_quench_yhi"],
    cov_ul_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ul_quench_ylo"],
    cov_ul_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ul_quench_yhi"],
    cov_utau_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_utau_quench_ylo"],
    cov_utau_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_utau_quench_yhi"],
    cov_uqt_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_uqt_quench_ylo"],
    cov_uqt_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_uqt_quench_yhi"],
    cov_uqs_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqs_quench_ylo"],
    cov_uqs_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqs_quench_yhi"],
    cov_udrop_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_udrop_quench_ylo"
    ],
    cov_udrop_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_udrop_quench_yhi"
    ],
    cov_urej_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_urej_quench_ylo"],
    cov_urej_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_urej_quench_yhi"],
    cov_ulgy_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgm_quench_ylo"],
    cov_ulgy_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgm_quench_yhi"],
    cov_ul_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgm_quench_ylo"],
    cov_ul_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgm_quench_yhi"],
    cov_ul_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgy_quench_ylo"],
    cov_ul_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgy_quench_yhi"],
    cov_utau_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgm_quench_ylo"],
    cov_utau_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgm_quench_yhi"],
    cov_utau_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgy_quench_ylo"],
    cov_utau_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgy_quench_yhi"],
    cov_utau_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ul_quench_ylo"],
    cov_utau_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ul_quench_yhi"],
    cov_uqt_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgm_quench_ylo"],
    cov_uqt_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgm_quench_yhi"],
    cov_uqt_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgy_quench_ylo"],
    cov_uqt_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgy_quench_yhi"],
    cov_uqt_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ul_quench_ylo"],
    cov_uqt_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ul_quench_yhi"],
    cov_uqt_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_utau_quench_ylo"],
    cov_uqt_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_utau_quench_yhi"],
    cov_uqs_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgm_quench_ylo"],
    cov_uqs_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgm_quench_yhi"],
    cov_uqs_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgy_quench_ylo"],
    cov_uqs_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgy_quench_yhi"],
    cov_uqs_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ul_quench_ylo"],
    cov_uqs_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ul_quench_yhi"],
    cov_uqs_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_utau_quench_ylo"],
    cov_uqs_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_utau_quench_yhi"],
    cov_uqs_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqt_quench_ylo"],
    cov_uqs_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqt_quench_yhi"],
    cov_udrop_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgm_quench_ylo"
    ],
    cov_udrop_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgm_quench_yhi"
    ],
    cov_udrop_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgy_quench_ylo"
    ],
    cov_udrop_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgy_quench_yhi"
    ],
    cov_udrop_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_ul_quench_ylo"],
    cov_udrop_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_ul_quench_yhi"],
    cov_udrop_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_utau_quench_ylo"
    ],
    cov_udrop_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_utau_quench_yhi"
    ],
    cov_udrop_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqt_quench_ylo"],
    cov_udrop_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqt_quench_yhi"],
    cov_udrop_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqs_quench_ylo"],
    cov_udrop_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqs_quench_yhi"],
    cov_urej_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgm_quench_ylo"],
    cov_urej_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgm_quench_yhi"],
    cov_urej_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgy_quench_ylo"],
    cov_urej_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgy_quench_yhi"],
    cov_urej_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ul_quench_ylo"],
    cov_urej_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ul_quench_yhi"],
    cov_urej_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_utau_quench_ylo"],
    cov_urej_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_utau_quench_yhi"],
    cov_urej_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqt_quench_ylo"],
    cov_urej_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqt_quench_yhi"],
    cov_urej_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqs_quench_ylo"],
    cov_urej_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqs_quench_yhi"],
    cov_urej_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_urej_udrop_quench_ylo"
    ],
    cov_urej_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_urej_udrop_quench_yhi"
    ],
):

    frac_quench = frac_quench_vs_lgm0(
        logmp_arr, frac_quench_x0, frac_quench_k, frac_quench_ylo, frac_quench_yhi
    )
    _res = _get_mean_smah_params_quench(
        logmp_arr,
        mean_ulgm_quench_ylo,
        mean_ulgm_quench_yhi,
        mean_ulgy_quench_ylo,
        mean_ulgy_quench_yhi,
        mean_ul_quench_ylo,
        mean_ul_quench_yhi,
        mean_utau_quench_ylo,
        mean_utau_quench_yhi,
        mean_uqt_quench_ylo,
        mean_uqt_quench_yhi,
        mean_uqs_quench_ylo,
        mean_uqs_quench_yhi,
        mean_udrop_quench_ylo,
        mean_udrop_quench_yhi,
        mean_urej_quench_ylo,
        mean_urej_quench_yhi,
    )

    means_quench = jnp.array(_res).T

    covs_quench = _get_covs_quench(
        logmp_arr,
        cov_ulgm_ulgm_quench_ylo,
        cov_ulgm_ulgm_quench_yhi,
        cov_ulgy_ulgy_quench_ylo,
        cov_ulgy_ulgy_quench_yhi,
        cov_ul_ul_quench_ylo,
        cov_ul_ul_quench_yhi,
        cov_utau_utau_quench_ylo,
        cov_utau_utau_quench_yhi,
        cov_uqt_uqt_quench_ylo,
        cov_uqt_uqt_quench_yhi,
        cov_uqs_uqs_quench_ylo,
        cov_uqs_uqs_quench_yhi,
        cov_udrop_udrop_quench_ylo,
        cov_udrop_udrop_quench_yhi,
        cov_urej_urej_quench_ylo,
        cov_urej_urej_quench_yhi,
        cov_ulgy_ulgm_quench_ylo,
        cov_ulgy_ulgm_quench_yhi,
        cov_ul_ulgm_quench_ylo,
        cov_ul_ulgm_quench_yhi,
        cov_ul_ulgy_quench_ylo,
        cov_ul_ulgy_quench_yhi,
        cov_utau_ulgm_quench_ylo,
        cov_utau_ulgm_quench_yhi,
        cov_utau_ulgy_quench_ylo,
        cov_utau_ulgy_quench_yhi,
        cov_utau_ul_quench_ylo,
        cov_utau_ul_quench_yhi,
        cov_uqt_ulgm_quench_ylo,
        cov_uqt_ulgm_quench_yhi,
        cov_uqt_ulgy_quench_ylo,
        cov_uqt_ulgy_quench_yhi,
        cov_uqt_ul_quench_ylo,
        cov_uqt_ul_quench_yhi,
        cov_uqt_utau_quench_ylo,
        cov_uqt_utau_quench_yhi,
        cov_uqs_ulgm_quench_ylo,
        cov_uqs_ulgm_quench_yhi,
        cov_uqs_ulgy_quench_ylo,
        cov_uqs_ulgy_quench_yhi,
        cov_uqs_ul_quench_ylo,
        cov_uqs_ul_quench_yhi,
        cov_uqs_utau_quench_ylo,
        cov_uqs_utau_quench_yhi,
        cov_uqs_uqt_quench_ylo,
        cov_uqs_uqt_quench_yhi,
        cov_udrop_ulgm_quench_ylo,
        cov_udrop_ulgm_quench_yhi,
        cov_udrop_ulgy_quench_ylo,
        cov_udrop_ulgy_quench_yhi,
        cov_udrop_ul_quench_ylo,
        cov_udrop_ul_quench_yhi,
        cov_udrop_utau_quench_ylo,
        cov_udrop_utau_quench_yhi,
        cov_udrop_uqt_quench_ylo,
        cov_udrop_uqt_quench_yhi,
        cov_udrop_uqs_quench_ylo,
        cov_udrop_uqs_quench_yhi,
        cov_urej_ulgm_quench_ylo,
        cov_urej_ulgm_quench_yhi,
        cov_urej_ulgy_quench_ylo,
        cov_urej_ulgy_quench_yhi,
        cov_urej_ul_quench_ylo,
        cov_urej_ul_quench_yhi,
        cov_urej_utau_quench_ylo,
        cov_urej_utau_quench_yhi,
        cov_urej_uqt_quench_ylo,
        cov_urej_uqt_quench_yhi,
        cov_urej_uqs_quench_ylo,
        cov_urej_uqs_quench_yhi,
        cov_urej_udrop_quench_ylo,
        cov_urej_udrop_quench_yhi,
    )
    return frac_quench, means_quench, covs_quench


@jjit
def _get_mean_smah_params_quench(
    lgm,
    mean_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgm_quench_ylo"],
    mean_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgm_quench_yhi"],
    mean_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgy_quench_ylo"],
    mean_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ulgy_quench_yhi"],
    mean_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ul_quench_ylo"],
    mean_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_ul_quench_yhi"],
    mean_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_utau_quench_ylo"],
    mean_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_utau_quench_yhi"],
    mean_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqt_quench_ylo"],
    mean_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqt_quench_yhi"],
    mean_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqs_quench_ylo"],
    mean_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_uqs_quench_yhi"],
    mean_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_udrop_quench_ylo"],
    mean_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_udrop_quench_yhi"],
    mean_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_urej_quench_ylo"],
    mean_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["mean_urej_quench_yhi"],
):
    ulgm = mean_ulgm_quench_vs_lgm0(lgm, mean_ulgm_quench_ylo, mean_ulgm_quench_yhi)
    ulgy = mean_ulgy_quench_vs_lgm0(lgm, mean_ulgy_quench_ylo, mean_ulgy_quench_yhi)
    ul = mean_ul_quench_vs_lgm0(lgm, mean_ul_quench_ylo, mean_ul_quench_yhi)
    utau = mean_utau_quench_vs_lgm0(lgm, mean_utau_quench_ylo, mean_utau_quench_yhi)
    uqt = mean_uqt_quench_vs_lgm0(lgm, mean_uqt_quench_ylo, mean_uqt_quench_yhi)
    uqs = mean_uqs_quench_vs_lgm0(lgm, mean_uqs_quench_ylo, mean_uqs_quench_yhi)
    udrop = mean_udrop_quench_vs_lgm0(lgm, mean_udrop_quench_ylo, mean_udrop_quench_yhi)
    urej = mean_urej_quench_vs_lgm0(lgm, mean_urej_quench_ylo, mean_urej_quench_yhi)
    return ulgm, ulgy, ul, utau, uqt, uqs, udrop, urej


@jjit
def _get_covs_quench(
    lgmp_arr,
    cov_ulgm_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgm_ulgm_quench_ylo"],
    cov_ulgm_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgm_ulgm_quench_yhi"],
    cov_ulgy_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgy_quench_ylo"],
    cov_ulgy_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgy_quench_yhi"],
    cov_ul_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ul_quench_ylo"],
    cov_ul_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ul_quench_yhi"],
    cov_utau_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_utau_quench_ylo"],
    cov_utau_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_utau_quench_yhi"],
    cov_uqt_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_uqt_quench_ylo"],
    cov_uqt_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_uqt_quench_yhi"],
    cov_uqs_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqs_quench_ylo"],
    cov_uqs_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqs_quench_yhi"],
    cov_udrop_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_udrop_quench_ylo"
    ],
    cov_udrop_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_udrop_quench_yhi"
    ],
    cov_urej_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_urej_quench_ylo"],
    cov_urej_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_urej_quench_yhi"],
    cov_ulgy_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgm_quench_ylo"],
    cov_ulgy_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgm_quench_yhi"],
    cov_ul_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgm_quench_ylo"],
    cov_ul_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgm_quench_yhi"],
    cov_ul_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgy_quench_ylo"],
    cov_ul_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgy_quench_yhi"],
    cov_utau_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgm_quench_ylo"],
    cov_utau_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgm_quench_yhi"],
    cov_utau_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgy_quench_ylo"],
    cov_utau_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgy_quench_yhi"],
    cov_utau_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ul_quench_ylo"],
    cov_utau_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ul_quench_yhi"],
    cov_uqt_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgm_quench_ylo"],
    cov_uqt_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgm_quench_yhi"],
    cov_uqt_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgy_quench_ylo"],
    cov_uqt_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgy_quench_yhi"],
    cov_uqt_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ul_quench_ylo"],
    cov_uqt_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ul_quench_yhi"],
    cov_uqt_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_utau_quench_ylo"],
    cov_uqt_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_utau_quench_yhi"],
    cov_uqs_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgm_quench_ylo"],
    cov_uqs_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgm_quench_yhi"],
    cov_uqs_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgy_quench_ylo"],
    cov_uqs_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgy_quench_yhi"],
    cov_uqs_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ul_quench_ylo"],
    cov_uqs_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ul_quench_yhi"],
    cov_uqs_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_utau_quench_ylo"],
    cov_uqs_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_utau_quench_yhi"],
    cov_uqs_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqt_quench_ylo"],
    cov_uqs_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqt_quench_yhi"],
    cov_udrop_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgm_quench_ylo"
    ],
    cov_udrop_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgm_quench_yhi"
    ],
    cov_udrop_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgy_quench_ylo"
    ],
    cov_udrop_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgy_quench_yhi"
    ],
    cov_udrop_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_ul_quench_ylo"],
    cov_udrop_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_ul_quench_yhi"],
    cov_udrop_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_utau_quench_ylo"
    ],
    cov_udrop_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_utau_quench_yhi"
    ],
    cov_udrop_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqt_quench_ylo"],
    cov_udrop_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqt_quench_yhi"],
    cov_udrop_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqs_quench_ylo"],
    cov_udrop_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqs_quench_yhi"],
    cov_urej_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgm_quench_ylo"],
    cov_urej_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgm_quench_yhi"],
    cov_urej_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgy_quench_ylo"],
    cov_urej_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgy_quench_yhi"],
    cov_urej_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ul_quench_ylo"],
    cov_urej_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ul_quench_yhi"],
    cov_urej_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_utau_quench_ylo"],
    cov_urej_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_utau_quench_yhi"],
    cov_urej_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqt_quench_ylo"],
    cov_urej_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqt_quench_yhi"],
    cov_urej_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqs_quench_ylo"],
    cov_urej_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqs_quench_yhi"],
    cov_urej_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_urej_udrop_quench_ylo"
    ],
    cov_urej_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_urej_udrop_quench_yhi"
    ],
):

    _res = _get_cov_params_quench(
        lgmp_arr,
        cov_ulgm_ulgm_quench_ylo,
        cov_ulgm_ulgm_quench_yhi,
        cov_ulgy_ulgy_quench_ylo,
        cov_ulgy_ulgy_quench_yhi,
        cov_ul_ul_quench_ylo,
        cov_ul_ul_quench_yhi,
        cov_utau_utau_quench_ylo,
        cov_utau_utau_quench_yhi,
        cov_uqt_uqt_quench_ylo,
        cov_uqt_uqt_quench_yhi,
        cov_uqs_uqs_quench_ylo,
        cov_uqs_uqs_quench_yhi,
        cov_udrop_udrop_quench_ylo,
        cov_udrop_udrop_quench_yhi,
        cov_urej_urej_quench_ylo,
        cov_urej_urej_quench_yhi,
        cov_ulgy_ulgm_quench_ylo,
        cov_ulgy_ulgm_quench_yhi,
        cov_ul_ulgm_quench_ylo,
        cov_ul_ulgm_quench_yhi,
        cov_ul_ulgy_quench_ylo,
        cov_ul_ulgy_quench_yhi,
        cov_utau_ulgm_quench_ylo,
        cov_utau_ulgm_quench_yhi,
        cov_utau_ulgy_quench_ylo,
        cov_utau_ulgy_quench_yhi,
        cov_utau_ul_quench_ylo,
        cov_utau_ul_quench_yhi,
        cov_uqt_ulgm_quench_ylo,
        cov_uqt_ulgm_quench_yhi,
        cov_uqt_ulgy_quench_ylo,
        cov_uqt_ulgy_quench_yhi,
        cov_uqt_ul_quench_ylo,
        cov_uqt_ul_quench_yhi,
        cov_uqt_utau_quench_ylo,
        cov_uqt_utau_quench_yhi,
        cov_uqs_ulgm_quench_ylo,
        cov_uqs_ulgm_quench_yhi,
        cov_uqs_ulgy_quench_ylo,
        cov_uqs_ulgy_quench_yhi,
        cov_uqs_ul_quench_ylo,
        cov_uqs_ul_quench_yhi,
        cov_uqs_utau_quench_ylo,
        cov_uqs_utau_quench_yhi,
        cov_uqs_uqt_quench_ylo,
        cov_uqs_uqt_quench_yhi,
        cov_udrop_ulgm_quench_ylo,
        cov_udrop_ulgm_quench_yhi,
        cov_udrop_ulgy_quench_ylo,
        cov_udrop_ulgy_quench_yhi,
        cov_udrop_ul_quench_ylo,
        cov_udrop_ul_quench_yhi,
        cov_udrop_utau_quench_ylo,
        cov_udrop_utau_quench_yhi,
        cov_udrop_uqt_quench_ylo,
        cov_udrop_uqt_quench_yhi,
        cov_udrop_uqs_quench_ylo,
        cov_udrop_uqs_quench_yhi,
        cov_urej_ulgm_quench_ylo,
        cov_urej_ulgm_quench_yhi,
        cov_urej_ulgy_quench_ylo,
        cov_urej_ulgy_quench_yhi,
        cov_urej_ul_quench_ylo,
        cov_urej_ul_quench_yhi,
        cov_urej_utau_quench_ylo,
        cov_urej_utau_quench_yhi,
        cov_urej_uqt_quench_ylo,
        cov_urej_uqt_quench_yhi,
        cov_urej_uqs_quench_ylo,
        cov_urej_uqs_quench_yhi,
        cov_urej_udrop_quench_ylo,
        cov_urej_udrop_quench_yhi,
    )
    return _get_cov_vmap(*_res)


@jjit
def _get_cov_params_quench(
    lgm,
    cov_ulgm_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgm_ulgm_quench_ylo"],
    cov_ulgm_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgm_ulgm_quench_yhi"],
    cov_ulgy_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgy_quench_ylo"],
    cov_ulgy_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgy_quench_yhi"],
    cov_ul_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ul_quench_ylo"],
    cov_ul_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ul_quench_yhi"],
    cov_utau_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_utau_quench_ylo"],
    cov_utau_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_utau_quench_yhi"],
    cov_uqt_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_uqt_quench_ylo"],
    cov_uqt_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_uqt_quench_yhi"],
    cov_uqs_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqs_quench_ylo"],
    cov_uqs_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqs_quench_yhi"],
    cov_udrop_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_udrop_quench_ylo"
    ],
    cov_udrop_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_udrop_quench_yhi"
    ],
    cov_urej_urej_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_urej_quench_ylo"],
    cov_urej_urej_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_urej_quench_yhi"],
    cov_ulgy_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgm_quench_ylo"],
    cov_ulgy_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ulgy_ulgm_quench_yhi"],
    cov_ul_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgm_quench_ylo"],
    cov_ul_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgm_quench_yhi"],
    cov_ul_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgy_quench_ylo"],
    cov_ul_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_ul_ulgy_quench_yhi"],
    cov_utau_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgm_quench_ylo"],
    cov_utau_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgm_quench_yhi"],
    cov_utau_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgy_quench_ylo"],
    cov_utau_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ulgy_quench_yhi"],
    cov_utau_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ul_quench_ylo"],
    cov_utau_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_utau_ul_quench_yhi"],
    cov_uqt_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgm_quench_ylo"],
    cov_uqt_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgm_quench_yhi"],
    cov_uqt_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgy_quench_ylo"],
    cov_uqt_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ulgy_quench_yhi"],
    cov_uqt_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ul_quench_ylo"],
    cov_uqt_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_ul_quench_yhi"],
    cov_uqt_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_utau_quench_ylo"],
    cov_uqt_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqt_utau_quench_yhi"],
    cov_uqs_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgm_quench_ylo"],
    cov_uqs_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgm_quench_yhi"],
    cov_uqs_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgy_quench_ylo"],
    cov_uqs_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ulgy_quench_yhi"],
    cov_uqs_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ul_quench_ylo"],
    cov_uqs_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_ul_quench_yhi"],
    cov_uqs_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_utau_quench_ylo"],
    cov_uqs_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_utau_quench_yhi"],
    cov_uqs_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqt_quench_ylo"],
    cov_uqs_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_uqs_uqt_quench_yhi"],
    cov_udrop_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgm_quench_ylo"
    ],
    cov_udrop_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgm_quench_yhi"
    ],
    cov_udrop_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgy_quench_ylo"
    ],
    cov_udrop_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_ulgy_quench_yhi"
    ],
    cov_udrop_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_ul_quench_ylo"],
    cov_udrop_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_ul_quench_yhi"],
    cov_udrop_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_utau_quench_ylo"
    ],
    cov_udrop_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_udrop_utau_quench_yhi"
    ],
    cov_udrop_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqt_quench_ylo"],
    cov_udrop_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqt_quench_yhi"],
    cov_udrop_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqs_quench_ylo"],
    cov_udrop_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_udrop_uqs_quench_yhi"],
    cov_urej_ulgm_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgm_quench_ylo"],
    cov_urej_ulgm_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgm_quench_yhi"],
    cov_urej_ulgy_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgy_quench_ylo"],
    cov_urej_ulgy_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ulgy_quench_yhi"],
    cov_urej_ul_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ul_quench_ylo"],
    cov_urej_ul_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_ul_quench_yhi"],
    cov_urej_utau_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_utau_quench_ylo"],
    cov_urej_utau_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_utau_quench_yhi"],
    cov_urej_uqt_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqt_quench_ylo"],
    cov_urej_uqt_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqt_quench_yhi"],
    cov_urej_uqs_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqs_quench_ylo"],
    cov_urej_uqs_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS["cov_urej_uqs_quench_yhi"],
    cov_urej_udrop_quench_ylo=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_urej_udrop_quench_ylo"
    ],
    cov_urej_udrop_quench_yhi=DEFAULT_SFH_PDF_QUENCH_PARAMS[
        "cov_urej_udrop_quench_yhi"
    ],
):
    ulgm_ulgm = cov_ulgm_ulgm_quench_vs_lgm0(
        lgm, cov_ulgm_ulgm_quench_ylo, cov_ulgm_ulgm_quench_yhi
    )
    ulgy_ulgy = cov_ulgy_ulgy_quench_vs_lgm0(
        lgm, cov_ulgy_ulgy_quench_ylo, cov_ulgy_ulgy_quench_yhi
    )
    ul_ul = cov_ul_ul_quench_vs_lgm0(lgm, cov_ul_ul_quench_ylo, cov_ul_ul_quench_yhi)
    utau_utau = cov_utau_utau_quench_vs_lgm0(
        lgm, cov_utau_utau_quench_ylo, cov_utau_utau_quench_yhi
    )
    uqt_uqt = cov_uqt_uqt_quench_vs_lgm0(
        lgm, cov_uqt_uqt_quench_ylo, cov_uqt_uqt_quench_yhi
    )
    uqs_uqs = cov_uqs_uqs_quench_vs_lgm0(
        lgm, cov_uqs_uqs_quench_ylo, cov_uqs_uqs_quench_yhi
    )
    udrop_udrop = cov_udrop_udrop_quench_vs_lgm0(
        lgm, cov_udrop_udrop_quench_ylo, cov_udrop_udrop_quench_yhi
    )
    urej_urej = cov_urej_urej_quench_vs_lgm0(
        lgm, cov_urej_urej_quench_ylo, cov_urej_urej_quench_yhi
    )
    ulgy_ulgm = cov_ulgy_ulgm_quench_vs_lgm0(
        lgm, cov_ulgy_ulgm_quench_ylo, cov_ulgy_ulgm_quench_yhi
    )
    ul_ulgm = cov_ul_ulgm_quench_vs_lgm0(
        lgm, cov_ul_ulgm_quench_ylo, cov_ul_ulgm_quench_yhi
    )
    ul_ulgy = cov_ul_ulgy_quench_vs_lgm0(
        lgm, cov_ul_ulgy_quench_ylo, cov_ul_ulgy_quench_yhi
    )
    utau_ulgm = cov_utau_ulgm_quench_vs_lgm0(
        lgm, cov_utau_ulgm_quench_ylo, cov_utau_ulgm_quench_yhi
    )
    utau_ulgy = cov_utau_ulgy_quench_vs_lgm0(
        lgm, cov_utau_ulgy_quench_ylo, cov_utau_ulgy_quench_yhi
    )
    utau_ul = cov_utau_ul_quench_vs_lgm0(
        lgm, cov_utau_ul_quench_ylo, cov_utau_ul_quench_yhi
    )
    uqt_ulgm = cov_uqt_ulgm_quench_vs_lgm0(
        lgm, cov_uqt_ulgm_quench_ylo, cov_uqt_ulgm_quench_yhi
    )
    uqt_ulgy = cov_uqt_ulgy_quench_vs_lgm0(
        lgm, cov_uqt_ulgy_quench_ylo, cov_uqt_ulgy_quench_yhi
    )
    uqt_ul = cov_uqt_ul_quench_vs_lgm0(
        lgm, cov_uqt_ul_quench_ylo, cov_uqt_ul_quench_yhi
    )
    uqt_utau = cov_uqt_utau_quench_vs_lgm0(
        lgm, cov_uqt_utau_quench_ylo, cov_uqt_utau_quench_yhi
    )
    uqs_ulgm = cov_uqs_ulgm_quench_vs_lgm0(
        lgm, cov_uqs_ulgm_quench_ylo, cov_uqs_ulgm_quench_yhi
    )
    uqs_ulgy = cov_uqs_ulgy_quench_vs_lgm0(
        lgm, cov_uqs_ulgy_quench_ylo, cov_uqs_ulgy_quench_yhi
    )
    uqs_ul = cov_uqs_ul_quench_vs_lgm0(
        lgm, cov_uqs_ul_quench_ylo, cov_uqs_ul_quench_yhi
    )
    uqs_utau = cov_uqs_utau_quench_vs_lgm0(
        lgm, cov_uqs_utau_quench_ylo, cov_uqs_utau_quench_yhi
    )
    uqs_uqt = cov_uqs_uqt_quench_vs_lgm0(
        lgm, cov_uqs_uqt_quench_ylo, cov_uqs_uqt_quench_yhi
    )
    udrop_ulgm = cov_udrop_ulgm_quench_vs_lgm0(
        lgm, cov_udrop_ulgm_quench_ylo, cov_udrop_ulgm_quench_yhi
    )
    udrop_ulgy = cov_udrop_ulgy_quench_vs_lgm0(
        lgm, cov_udrop_ulgy_quench_ylo, cov_udrop_ulgy_quench_yhi
    )
    udrop_ul = cov_udrop_ul_quench_vs_lgm0(
        lgm, cov_udrop_ul_quench_ylo, cov_udrop_ul_quench_yhi
    )
    udrop_utau = cov_udrop_utau_quench_vs_lgm0(
        lgm, cov_udrop_utau_quench_ylo, cov_udrop_utau_quench_yhi
    )
    udrop_uqt = cov_udrop_uqt_quench_vs_lgm0(
        lgm, cov_udrop_uqt_quench_ylo, cov_udrop_uqt_quench_yhi
    )
    udrop_uqs = cov_udrop_uqs_quench_vs_lgm0(
        lgm, cov_udrop_uqs_quench_ylo, cov_udrop_uqs_quench_yhi
    )
    urej_ulgm = cov_urej_ulgm_quench_vs_lgm0(
        lgm, cov_urej_ulgm_quench_ylo, cov_urej_ulgm_quench_yhi
    )
    urej_ulgy = cov_urej_ulgy_quench_vs_lgm0(
        lgm, cov_urej_ulgy_quench_ylo, cov_urej_ulgy_quench_yhi
    )
    urej_ul = cov_urej_ul_quench_vs_lgm0(
        lgm, cov_urej_ul_quench_ylo, cov_urej_ul_quench_yhi
    )
    urej_utau = cov_urej_utau_quench_vs_lgm0(
        lgm, cov_urej_utau_quench_ylo, cov_urej_utau_quench_yhi
    )
    urej_uqt = cov_urej_uqt_quench_vs_lgm0(
        lgm, cov_urej_uqt_quench_ylo, cov_urej_uqt_quench_yhi
    )
    urej_uqs = cov_urej_uqs_quench_vs_lgm0(
        lgm, cov_urej_uqs_quench_ylo, cov_urej_uqs_quench_yhi
    )
    urej_udrop = cov_urej_udrop_quench_vs_lgm0(
        lgm, cov_urej_udrop_quench_ylo, cov_urej_udrop_quench_yhi
    )

    cov_params = (
        ulgm_ulgm,
        ulgy_ulgy,
        ul_ul,
        utau_utau,
        uqt_uqt,
        uqs_uqs,
        udrop_udrop,
        urej_urej,
        ulgy_ulgm,
        ul_ulgm,
        ul_ulgy,
        utau_ulgm,
        utau_ulgy,
        utau_ul,
        uqt_ulgm,
        uqt_ulgy,
        uqt_ul,
        uqt_utau,
        uqs_ulgm,
        uqs_ulgy,
        uqs_ul,
        uqs_utau,
        uqs_uqt,
        udrop_ulgm,
        udrop_ulgy,
        udrop_ul,
        udrop_utau,
        udrop_uqt,
        udrop_uqs,
        urej_ulgm,
        urej_ulgy,
        urej_ul,
        urej_utau,
        urej_uqt,
        urej_uqs,
        urej_udrop,
    )

    return cov_params
