"""
"""
import numpy as np

from ..pecZ import pecZ

EPS = 1e-3


def test_pecZ():
    n = 100
    Lbox = 1_000
    x = np.random.uniform(0, Lbox, n)
    y = np.random.uniform(0, Lbox, n)
    z = np.random.uniform(0, Lbox, n)
    vx = np.random.uniform(-100, 100, n)
    vy = np.random.uniform(-100, 100, n)
    vz = np.random.uniform(-100, 100, n)
    z_hubb = np.random.uniform(0, 3, n)
    _res = pecZ(x, y, z, vx, vy, vz, z_hubb)
    for _x in _res:
        assert np.all(np.isfinite(_x))
        assert _x.shape == (n,)
    z_pec, z_tot, v_pec, v_peca, r_rel_mag, r_rel_maga, r_dist = _res
    assert np.all(z_pec > -EPS)
    assert np.all(z_tot > -EPS)
