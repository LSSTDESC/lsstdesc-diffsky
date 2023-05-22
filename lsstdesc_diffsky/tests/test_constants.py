"""
"""
from .. import constants


def test_diffmah_pnames():
    assert constants.MAH_PNAMES == [
        "diffmah_logmp_fit",
        "diffmah_mah_logtc",
        "diffmah_early_index",
        "diffmah_late_index",
    ]


def test_diffstar_pnames():
    assert constants.MS_U_PNAMES == [
        "diffstar_u_lgmcrit",
        "diffstar_u_lgy_at_mcrit",
        "diffstar_u_indx_lo",
        "diffstar_u_indx_hi",
        "diffstar_u_tau_dep",
    ]
    assert constants.Q_U_PNAMES == [
        "diffstar_u_qt",
        "diffstar_u_qs",
        "diffstar_u_q_drop",
        "diffstar_u_q_rejuv",
    ]


def test_SED_params_contents():
    assert constants.SED_params == { 
        'mah_keys': constants.MAH_PNAMES,
        'ms_keys': constants.MS_U_PNAMES,
        'q_keys': constants.Q_U_PNAMES,
        'sfh_keys': ['mstar', 'sfr', 'fstar', 'dmhdt', 'log_mah'],
        'z0': 0.,
        't_start': .05,
        'N_t': 100,
        'xkeys': ['ssp_z_table', 'ssp_restmag_table', 'ssp_obsmag_table',
                  'ssp_lgmet', 'ssp_lg_age_gyr', 'filter_keys',
                  'filter_waves', 'filter_trans', 'met_params'],
    }
