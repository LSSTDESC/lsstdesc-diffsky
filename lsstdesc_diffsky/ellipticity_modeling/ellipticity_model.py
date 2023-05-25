"""
"""
import numpy as np
from scipy.stats import johnsonsb
from astropy.utils.misc import NumpyRNGContext
from halotools.utils import rank_order_percentile


__all__ = ("monte_carlo_ellipticity_bulge_disk",)


def monte_carlo_ellipticity_bulge_disk(magr, seed=None):
    """
    Model galaxy ellipticities using emprical model derived from observed data

    Parameters
    ----------
    magr: ndarray
        Numpy array of shape (ngals, )

    Returns
    -------
    ellip_bulge: ndarray
        Numpy array of shape (ngals, )

    ellip_disk: ndarray
        Numpy array of shape (ngals, )

    """

    magr = np.atleast_1d(magr)

    a_disk = np.interp(magr, [-21, -19], [-0.4, -0.4])
    # a_disk = calculate_johnsonsb_params_disk(mag_r)
    b_disk = np.ones_like(a_disk) * 0.7

    a_bulge = np.interp(magr, [-21, -19, -17], [0.6, 1.0, 1.6])
    # a_bulge = calculate_johnsonsb_params_bulge(mag_r)
    b_bulge = np.interp(magr, [-19, -17], [1.0, 1.0])
    # b_bulge = np.ones_like(a_bulge)

    with NumpyRNGContext(seed):
        urand = np.random.uniform(size=magr.size)
        urand2 = rank_order_percentile(
            1 * urand + 0.6 * np.random.uniform(size=magr.size)
        )
        ellip_bulge = johnsonsb.isf(urand, a_bulge, b_bulge)
        ellip_disk = johnsonsb.isf(urand2, a_disk, b_disk)
    return ellip_bulge, ellip_disk
