"""
"""
import numpy as np

from ..load_sfh import retrieve_diffstar_data


def _create_fake_data(n_gal, keys):
    fake_data = dict()
    for i, key in enumerate(keys):
        fake_data[key] = np.zeros(n_gal) + i
    return fake_data


def test_load_diffstar_data():
    mah_pnames = ["a", "b"]
    ms_pnames = ["c", "d"]
    q_pnames = ["e", "f", "g"]
    n_gals = 100
    fake_galcat = _create_fake_data(n_gals, mah_pnames + ms_pnames + q_pnames)
    diffstar_data = retrieve_diffstar_data(fake_galcat, mah_pnames, ms_pnames, q_pnames)
    assert diffstar_data.mah_params.shape == (n_gals, 2)
    assert diffstar_data.ms_params.shape == (n_gals, 2)
    assert diffstar_data.q_params.shape == (n_gals, 3)
