"""This module stores globals used throughout the repository."""
MAH_PNAMES = [
    "diffmah_logmp_fit",
    "diffmah_mah_logtc",
    "diffmah_early_index",
    "diffmah_late_index",
]
MS_PNAMES = [
    "diffstar_lgmcrit",
    "diffstar_lgy_at_mcrit",
    "diffstar_indx_lo",
    "diffstar_indx_hi",
    "diffstar_tau_dep",
]
Q_PNAMES = [
    "diffstar_lg_qt",
    "diffstar_qlglgdt",
    "diffstar_lg_drop",
    "diffstar_lg_rejuv",
]
SED_params = {
    "mah_keys": MAH_PNAMES,
    "ms_keys": MS_PNAMES,
    "q_keys": Q_PNAMES,
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
