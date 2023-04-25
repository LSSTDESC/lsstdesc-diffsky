"""Functions used to load information about galaxy/halo mass assembly from diffsky mocks
"""
import numpy as np
import typing
from ..constants import MAH_PNAMES, MS_U_PNAMES, Q_U_PNAMES


class DiffstarData(typing.NamedTuple):
    """NamedTuple with info about SSPs, filter data and cosmology"""

    mah_params: np.ndarray
    ms_u_params: np.ndarray
    q_u_params: np.ndarray


def retrieve_diffstar_data(
    data, mah_pnames=MAH_PNAMES, ms_u_pnames=MS_U_PNAMES, q_u_pnames=Q_U_PNAMES
):
    """Retrieve mock galaxy parameters that encode halo and star formation history

    Parameters
    ----------
    data : dictionary or table of column data

    mah_pnames : list, optional
        Column names storing the diffmah parameters
        Default is set by lsstdesc_diffsky.constants.MAH_PNAMES

    ms_u_pnames : list, optional
        Column names storing the diffmah parameters
        Default is set by lsstdesc_diffsky.constants.MS_U_PNAMES

    q_u_pnames : list, optional
        Column names storing the diffmah parameters
        Default is set by lsstdesc_diffsky.constants.Q_U_PNAMES

    Returns
    -------
    namedtuple with the following entries:

        mah_params : ndarray of shape (n_gals, n_mah_pars)

        ms_u_params : ndarray of shape (n_gals, n_mah_pars)

        q_u_params : ndarray of shape (n_gals, n_mah_pars)

    """
    mah_params = np.array([data[key] for key in mah_pnames]).T
    ms_u_params = np.array([data[key] for key in ms_u_pnames]).T
    q_u_params = np.array([data[key] for key in q_u_pnames]).T
    return DiffstarData(mah_params, ms_u_params, q_u_params)
