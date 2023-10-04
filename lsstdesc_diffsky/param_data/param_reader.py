"""
"""
import os
from collections import OrderedDict

import numpy as np

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


def _read_pdict(drn, bn):
    pdict = OrderedDict()
    with open(os.path.join(drn, bn), "r") as f:
        for raw_line in f:
            key, val_string = raw_line.strip().split()
            pdict[key] = float(val_string)
    return pdict


def read_mock_param_dictionaries(mock_name):
    """Read the parameter dictionaries defining the model used to generate a mock

    Parameters
    ----------
    mock_name : string
        Nickname of the mock. Current options include "roman_rubin_2023"

    Returns
    -------
    all_pdicts : list
        List of parameter dictionaries defining the behavior of the mock

    """
    if mock_name == "roman_rubin_2023":
        drn = os.path.join(_THIS_DRNAME, "roman_rubin_2023")

        lgfburst_u_pdict = _read_pdict(drn, "lgfburst_u_params.txt")
        burstshapepop_u_pdict = _read_pdict(drn, "burstshape_u_params.txt")
        lgav_pop_u_pdict = _read_pdict(drn, "lgav_dust_u_params.txt")
        dust_delta_pop_u_pdict = _read_pdict(drn, "delta_dust_u_params.txt")
        fracuno_pop_u_pdict = _read_pdict(drn, "funo_dust_u_params.txt")
        met_pdict = _read_pdict(drn, "lgmet_params.txt")

    all_param_dictionaries = (
        lgfburst_u_pdict,
        burstshapepop_u_pdict,
        lgav_pop_u_pdict,
        dust_delta_pop_u_pdict,
        fracuno_pop_u_pdict,
        met_pdict,
    )
    return all_param_dictionaries


def read_mock_params(mock_name):
    """Read the parameter arrays defining the model used to generate a mock

    Parameters
    ----------
    mock_name : string
        Nickname of the mock. Current options include "roman_rubin_2023"

    Returns
    -------
    all_pdicts : list
        List of parameter arrays defining the behavior of the mock

    """
    all_pdicts = read_mock_param_dictionaries(mock_name)
    all_params = [np.array(list(pdict.values())) for pdict in all_pdicts]
    return all_params
