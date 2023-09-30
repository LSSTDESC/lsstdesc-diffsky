"""Functions used to load information about galaxy/halo mass assembly from diffsky mocks
"""
import typing

import numpy as np

from ..constants import MAH_PNAMES, MS_PNAMES, Q_PNAMES


class DiffstarData(typing.NamedTuple):
    """NamedTuple with info about SSPs, filter data and cosmology"""

    mah_params: np.ndarray
    ms_params: np.ndarray
    q_params: np.ndarray


def retrieve_diffstar_data(
    data, mah_pnames=MAH_PNAMES, ms_pnames=MS_PNAMES, q_pnames=Q_PNAMES
):
    """Retrieve mock galaxy parameters that encode halo and star formation history

    Parameters
    ----------
    data : dictionary or table of column data

    mah_pnames : list, optional
        Column names storing the diffmah parameters
        Default is set by lsstdesc_diffsky.constants.MAH_PNAMES

    ms_pnames : list, optional
        Column names storing the diffstar parameters
        Default is set by lsstdesc_diffsky.constants.MS_PNAMES

    q_pnames : list, optional
        Column names storing the diffstar parameters
        Default is set by lsstdesc_diffsky.constants.Q_PNAMES

    Returns
    -------
    namedtuple with the following entries:

        mah_params : ndarray of shape (n_gals, n_mah_pars)

        ms_params : ndarray of shape (n_gals, n_mah_pars)

        q_params : ndarray of shape (n_gals, n_mah_pars)

    """
    mah_params = np.array([data[key] for key in mah_pnames]).T
    ms_params = np.array([data[key] for key in ms_pnames]).T
    q_params = np.array([data[key] for key in q_pnames]).T
    return DiffstarData(mah_params, ms_params, q_params)
