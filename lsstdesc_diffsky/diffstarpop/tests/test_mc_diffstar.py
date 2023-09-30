"""
"""
import numpy as np
from diffstar import sfh_galpop
from diffstar.defaults import FB, LGT0, MS_PARAM_BOUNDS_PDICT, Q_PARAM_BOUNDS_PDICT
from jax import random as jran

from ..mc_diffstar import mc_diffstarpop


def test_mc_diffstar_has_correct_shape():
    """Test a large number of halos so that there are always both quenched and
    star-forming galaxies
    """
    ran_key = jran.PRNGKey(0)
    n_halos = 1000
    t_obs = 10.0
    logmh = np.linspace(10, 15, n_halos)
    galpop = mc_diffstarpop(ran_key, t_obs, logmh=logmh)
    mah_params, msk_is_quenched, ms_params, q_params = galpop
    assert mah_params.shape == (n_halos, 4)
    assert msk_is_quenched.shape == (n_halos,)
    assert ms_params.shape == (n_halos, 5)
    assert q_params.shape == (n_halos, 4)


def test_mc_diffstar_has_correct_shape2():
    """Test a very small number of halos so that sometimes galaxies are either
    all quenched or all star-forming
    """
    n_tests = 20
    n_halos = 2

    for itest in range(n_tests):
        ran_key = jran.PRNGKey(itest)
        mh_key, t_obs_key = jran.split(ran_key, 2)
        logmh = np.array(jran.uniform(mh_key, minval=5, maxval=16, shape=(n_halos,)))
        t_obs = float(jran.uniform(t_obs_key, minval=1, maxval=13.7, shape=(1,)))
        try:
            galpop = mc_diffstarpop(ran_key, t_obs, logmh=logmh)
        except AssertionError:
            msg = "t_obs = {0}\nlogmh = {1}"
            raise ValueError(msg.format(t_obs, logmh))
        mah_params, msk_is_quenched, ms_params, q_params = galpop
        assert mah_params.shape == (n_halos, 4)
        assert msk_is_quenched.shape == (n_halos,)
        assert ms_params.shape == (n_halos, 5)
        assert q_params.shape == (n_halos, 4)


def test_mc_diffstar_has_reasonable_sfhs():
    """Test a large number of halos so that there are always both quenched and
    star-forming galaxies.
    """
    ran_key = jran.PRNGKey(0)
    n_halos = 1000
    t_obs = 10.0
    logmh = np.zeros(n_halos) + 12
    galpop = mc_diffstarpop(ran_key, t_obs, logmh=logmh)
    mah_params, msk_is_quenched, ms_params, q_params = galpop

    n_t = 50
    tarr = np.linspace(1, 13.7, n_t)
    sfh = sfh_galpop(tarr, mah_params, ms_params, q_params, lgt0=LGT0, fb=FB)
    assert sfh.shape == (n_halos, n_t)
    assert np.all(np.isfinite(sfh))
    assert np.all(sfh > 0)

    sfh_q = sfh[msk_is_quenched]
    sfh_ms = sfh[~msk_is_quenched]

    mean_sfh_q = np.mean(sfh_q, axis=0)
    mean_sfh_ms = np.mean(sfh_ms, axis=0)
    assert np.all(mean_sfh_q[-10:] <= mean_sfh_ms[-10:])


def test_mc_diffstar_is_consistent_between_logmh_and_mah_params_inputs():
    """Enforce that the same results are returned when logmh and mah_params correspond
    to the same halo population
    """
    ran_key = jran.PRNGKey(10)
    n_halos = 1000
    t_obs = 10.0
    logmh = np.zeros(n_halos) + 12
    galpop = mc_diffstarpop(ran_key, t_obs, logmh=logmh)
    mah_params, msk_is_quenched, ms_params, q_params = galpop

    galpop = mc_diffstarpop(ran_key, t_obs, mah_params=mah_params)
    mah_params2, msk_is_quenched2, ms_params2, q_params2 = galpop

    assert np.allclose(mah_params, mah_params2, rtol=1e-4)
    assert np.allclose(msk_is_quenched, msk_is_quenched, rtol=1e-4)
    assert np.allclose(ms_params, ms_params, rtol=1e-4)
    assert np.allclose(q_params, q_params, rtol=1e-4)

    n_t = 50
    tarr = np.linspace(1, 13.7, n_t)
    sfh = sfh_galpop(tarr, mah_params, ms_params, q_params, lgt0=LGT0, fb=FB)
    assert sfh.shape == (n_halos, n_t)
    assert np.all(np.isfinite(sfh))
    assert np.all(sfh > 0)

    sfh_q = sfh[msk_is_quenched]
    sfh_ms = sfh[~msk_is_quenched]

    mean_sfh_q = np.mean(sfh_q, axis=0)
    mean_sfh_ms = np.mean(sfh_ms, axis=0)
    assert np.all(mean_sfh_q[-10:] <= mean_sfh_ms[-10:])


def test_mc_diffstar_works_when_tobs_equals_t0():
    """Enforce sensible results for DiffstarPop when t0=t_obs"""
    ran_key = jran.PRNGKey(10)
    n_halos = 1000
    t0 = 13.8
    t_obs = t0
    logmh = np.zeros(n_halos) + 12
    galpop = mc_diffstarpop(ran_key, t_obs, logmh=logmh)
    mah_params, msk_is_quenched, ms_params, q_params = galpop

    n_t = 50
    tarr = np.linspace(1, t0, n_t)
    sfh = sfh_galpop(tarr, mah_params, ms_params, q_params, lgt0=LGT0, fb=FB)
    assert sfh.shape == (n_halos, n_t)
    assert np.all(np.isfinite(sfh))
    assert np.all(sfh > 0)

    sfh_q = sfh[msk_is_quenched]
    sfh_ms = sfh[~msk_is_quenched]

    mean_sfh_q = np.mean(sfh_q, axis=0)
    mean_sfh_ms = np.mean(sfh_ms, axis=0)
    assert np.all(mean_sfh_q[-10:] <= mean_sfh_ms[-10:])
    assert np.all(mean_sfh_q[-10:] <= mean_sfh_ms[-10:])


def test_mc_diffstar_respects_param_bounds():
    ran_key = jran.PRNGKey(10)
    n_halos = 1000
    t0 = 13.8
    t_obs = t0
    logmh = np.zeros(n_halos) + 12
    galpop = mc_diffstarpop(ran_key, t_obs, logmh=logmh)
    mah_params, msk_is_quenched, ms_params, q_params = galpop

    pat = "diffstar parameter `{}` does not respect bounds set by MS_PARAM_BOUNDS_PDICT"
    for ip, key in enumerate(MS_PARAM_BOUNDS_PDICT.keys()):
        p = ms_params[:, ip]
        lo, hi = MS_PARAM_BOUNDS_PDICT[key]
        assert np.all(lo < p), pat.format(key)
        assert np.all(p < hi), pat.format(key)

    pat = "diffstar parameter `{}` does not respect bounds set by Q_PARAM_BOUNDS_PDICT"
    for ip, key in enumerate(Q_PARAM_BOUNDS_PDICT.keys()):
        p = q_params[:, ip]
        lo, hi = Q_PARAM_BOUNDS_PDICT[key]
        assert np.all(lo < p), pat.format(key)
        assert np.all(p < hi), pat.format(key)
