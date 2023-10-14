"""
"""
import os
import typing
from collections import OrderedDict

import numpy as np

_THIS_DRNAME = os.path.dirname(os.path.abspath(__file__))


class DiffskyPopParams(typing.NamedTuple):
    """NamedTuple storing parameters of a DiffskyPop"""

    lgfburst_u_params: np.float32
    burstshape_u_params: np.float32
    lgav_dust_u_params: np.float32
    delta_dust_u_params: np.float32
    funo_dust_u_params: np.float32
    lgmet_params: np.float32


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


def read_diffskypop_params(mock_name):
    """Read the parameter arrays defining the model used to generate a diffsky mock

    Parameters
    ----------
    mock_name : string
        Nickname of the mock. Current options include "roman_rubin_2023"

    Returns
    -------
    DiffskyPopParams : NamedTuple with the following fields

        lgfburst_pop_u_params : ndarray, shape (n_pars_lgfburst_pop, )
            Unbounded parameters controlling Fburst,
            which sets the fractional contribution of a recent burst
            to the smooth SFH of a galaxy. For typical values, see
            dspspop.lgfburstpop.DEFAULT_LGFBURST_U_PARAMS

        burstshapepop_u_params : ndarray, shape (n_pars_burstshape_pop, )
            Unbounded parameters controlling the distribution of stellar ages
            of stars formed in a recent burst. For typical values, see
            dspspop.burstshapepop.DEFAULT_BURSTSHAPE_U_PARAMS

        lgav_u_params : ndarray, shape (n_pars_lgav_pop, )
            Unbounded parameters controlling the distribution of dust parameter Av,
            the normalization of the attenuation curve at λ_V=5500 angstrom.
            For typical values, see
            dspspop.lgavpop.DEFAULT_LGAV_U_PARAMS

        dust_delta_u_params : ndarray, shape (n_pars_dust_delta_pop, )
            Unbounded parameters controlling the distribution of dust parameter δ,
            which modifies the power-law slope of the attenuation curve.
            For typical values, see
            dspspop.dust_deltapop.DEFAULT_DUST_DELTA_U_PARAMS

        fracuno_pop_u_params : ndarray, shape (n_pars_fracuno_pop, )
            Unbounded parameters controlling the fraction of sightlines
            unobscured by dust. For typical values,
            see dspspop.boris_dust.DEFAULT_U_PARAMS

        met_params : ndarray, shape (n_pars_met_pop, ), optional
            Parameters controlling the mass-metallicity scaling relation.
            For typical values, see dsps.metallicity.mzr.DEFAULT_MZR_PDICT
            mzr_params = met_params[:-1]
            lgmet_scatter = met_params[-1]

    """
    all_pdicts = read_mock_param_dictionaries(mock_name)
    all_params = [np.array(list(pdict.values())) for pdict in all_pdicts]
    return DiffskyPopParams(*all_params)
