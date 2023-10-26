"""
"""
import numpy as np

from ..photometry.get_SFH_from_params import get_log_safe_ssfr

Ngals = 1_000_000
Mstar = 1e10
SFR = 1e11


def test_get_log_safe_ssfr():
    # setup arrays
    mstar = np.array([Mstar] * Ngals)
    sfr = np.zeros(Ngals)
    # replace selected values with finite values
    sfr[0 : int(Ngals / 2)] = SFR
    log_ssfr = get_log_safe_ssfr(mstar, sfr)
    assert np.all(np.isfinite(log_ssfr))
    assert log_ssfr.shape == (Ngals,)
    assert np.all(log_ssfr[0 : int(Ngals / 2)] == 1)
    assert np.all(log_ssfr[int(Ngals / 2) :] < -7)
