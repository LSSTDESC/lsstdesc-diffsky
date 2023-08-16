"""
"""
import numpy as np

from ..disk_bulge_kernels import (
    DEFAULT_FBULGE_PARAMS,
    DEFAULT_T10,
    DEFAULT_T90,
    FBULGE_MAX,
    FBULGE_MIN,
    _bulge_fraction_vs_tform,
    _bulge_sfh,
    _get_params_from_u_params,
    _get_u_params_from_params,
    calc_tform_kern,
)


def test_bulge_sfh():
    nt = 100
    tarr = np.linspace(0.1, 13.8, nt)
    sfh = np.ones_like(tarr)
    _res = _bulge_sfh(tarr, sfh, DEFAULT_FBULGE_PARAMS)
    for x in _res:
        assert np.all(np.isfinite(x))
        assert x.shape == (nt,)
    smh, fbulge, sfh_bulge, smh_bulge, bth = _res


def test_bulge_fraction_vs_tform():
    nt = 100
    tarr = np.linspace(0.1, 13.8, nt)
    t10, t90 = 2.0, 10.0
    fbulge = _bulge_fraction_vs_tform(tarr, t10, t90, DEFAULT_FBULGE_PARAMS)
    assert np.all(np.isfinite(fbulge))
    assert np.all(fbulge > FBULGE_MIN)
    assert np.all(fbulge < FBULGE_MAX)


def test_param_bounding():
    n_tests = 1_000
    for __ in range(n_tests):
        u_params = np.random.uniform(-5, 5, len(DEFAULT_FBULGE_PARAMS))

        params = np.array(_get_params_from_u_params(u_params, DEFAULT_T10, DEFAULT_T90))
        assert np.all(np.isfinite(params))

        tcrit, frac_early, frac_late = params
        assert DEFAULT_T10 <= tcrit <= DEFAULT_T90
        assert FBULGE_MIN < frac_early < FBULGE_MAX
        assert FBULGE_MIN < frac_late < FBULGE_MAX
        assert frac_late < frac_early

        inferred_u_params = _get_u_params_from_params(params, DEFAULT_T10, DEFAULT_T90)
        assert np.all(np.isfinite(inferred_u_params)), params

        assert np.allclose(u_params, inferred_u_params, rtol=0.01)


def test_calc_tform_kern():
    tarr = np.linspace(0.1, 13.8, 200)
    smh = np.logspace(5, 12, tarr.size)
    t10 = calc_tform_kern(tarr, smh, 0.1)
    t90 = calc_tform_kern(tarr, smh, 0.9)
    assert t10 < t90
