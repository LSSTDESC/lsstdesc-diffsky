"""
"""
import numpy as np
from diffsky.experimental.dspspop.boris_dust import DEFAULT_PDICT as DEFAULT_FUNO_PDICT
from diffsky.experimental.dspspop.burstshapepop import DEFAULT_BURSTSHAPE_PDICT
from diffsky.experimental.dspspop.dust_deltapop import DEFAULT_DUST_DELTA_PDICT
from diffsky.experimental.dspspop.lgavpop import DEFAULT_LGAV_PDICT
from diffsky.experimental.dspspop.lgfburstpop import DEFAULT_LGFBURST_PDICT
from dsps.metallicity.mzr import DEFAULT_MET_PDICT

from ..param_reader import read_mock_param_dictionaries, read_mock_params


def test_read_mock_params_roman_rubin_2023():
    all_pdicts = read_mock_param_dictionaries("roman_rubin_2023")
    (
        lgfburst_u_pdict,
        burstshapepop_u_pdict,
        lgav_pop_u_pdict,
        dust_delta_pop_u_pdict,
        fracuno_pop_u_pdict,
        met_pdict,
    ) = all_pdicts

    assert set(DEFAULT_LGFBURST_PDICT.keys()) == set(lgfburst_u_pdict.keys())
    assert set(DEFAULT_BURSTSHAPE_PDICT.keys()) == set(burstshapepop_u_pdict.keys())
    assert set(DEFAULT_LGAV_PDICT.keys()) == set(lgav_pop_u_pdict.keys())
    assert set(DEFAULT_DUST_DELTA_PDICT.keys()) == set(dust_delta_pop_u_pdict.keys())
    assert set(DEFAULT_FUNO_PDICT.keys()) == set(fracuno_pop_u_pdict.keys())
    assert set(DEFAULT_MET_PDICT.keys()) == set(met_pdict.keys())


def test_read_mock_params():
    all_params = read_mock_params("roman_rubin_2023")
    all_pdicts = read_mock_param_dictionaries("roman_rubin_2023")
    for p, d in zip(all_params, all_pdicts):
        assert np.allclose(p, np.array(list(d.values())))
