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
    assert constants.MS_PNAMES == [
        "diffstar_lgmcrit",
        "diffstar_lgy_at_mcrit",
        "diffstar_indx_lo",
        "diffstar_indx_hi",
        "diffstar_tau_dep",
    ]
    assert constants.Q_PNAMES == [
        "diffstar_lg_qt",
        "diffstar_qlglgdt",
        "diffstar_lg_drop",
        "diffstar_lg_rejuv",
    ]


def test_bulge_pnames():
    assert constants.FBULGE_PNAMES == [
        "fbulge_tcrit",
        "fbulge_early",
        "fbulge_late",
    ]


def test_burstshape_pnames():
    assert constants.BURSTSHAPE_PNAMES == [
        "burstshape_lgyr_peak",
        "burstshape_lgyr_max",
    ]


def test_SED_params_contents():
    assert constants.SED_params == {
        "mah_keys": constants.MAH_PNAMES,
        "ms_keys": constants.MS_PNAMES,
        "q_keys": constants.Q_PNAMES,
        "sfh_keys": ["mstar", "sfr", "fstar", "dmhdt", "log_mah"],
        "z0": 0.0,
        "t_start": 0.05,
        "N_t": 100,
        "xkeys": [
            "ssp_z_table",
            "ssp_restmag_table",
            "ssp_obsmag_table",
            "ssp_lgmet",
            "ssp_lg_age_gyr",
            "filter_keys",
            "filter_waves",
            "filter_trans",
        ],
    }
