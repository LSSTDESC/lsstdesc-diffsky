import os
import glob
from astropy.table import Table, vstack
import numpy as np
from jax import numpy as jnp
from collections import OrderedDict


def read_pscan(pscan_dir, pscan_fname):
    pfiles = sorted(glob.glob(os.path.join(pscan_dir, pscan_fname)))

    first = True
    for n, pf in enumerate(pfiles):
        if first:
            pscan = Table.read(pf, format="ascii")
            first = False
        else:
            pscan = vstack([pscan, Table.read(pf, format="ascii")])

    print("Read {} pscan rows".format(len(pscan)))

    return pscan


def get_alt_dustpop_params_dict(pscan, row, dustpar_cols):
    pdict = OrderedDict()
    for pname in dustpar_cols:
        pdict[pname] = pscan[pname][row]

    print("Retrieved dust param values:\n", pdict)
    return pdict


def get_dustparams_key(pscan, row, dustparams_cols):
    dustparams = np.zeros(len(dustparams_cols), dtype=int)
    for n, pname in enumerate(dustparams_cols):
        dustparams[n] = pscan[pname][row]

    return jnp.array(dustparams, dtype=jnp.uint32)


def get_alt_dustpop_params(SED_params):
    assert "pscan_dirname" in SED_params.keys(), "Missing pscan_dirname"
    assert "pscan_filename" in SED_params.keys(), "Missing pscan_filename"
    assert "pscan_rownumber" in SED_params.keys(), "Missing pscan_rownumber"
    pscan = read_pscan(SED_params["pscan_dirname"], SED_params["pscan_filename"])
    row = SED_params["pscan_rownumber"]
    print(
        "...Retrieving alternate dust parameters from {}, row # {}".format(
            SED_params["pscan_filename"], row
        )
    )
    dustpar_cols = [c for c in pscan.colnames if "taueff" in c or "zboost" in c]
    alt_dustpop_params = get_alt_dustpop_params_dict(pscan, row, dustpar_cols)
    SED_params["alt_dustpop_params"] = alt_dustpop_params
    dustparams_cols = [c for c in pscan.colnames if "Dustparam" in c]
    dustparams_key = get_dustparams_key(pscan, row, dustparams_cols)
    SED_params["dustparams_key"] = dustparams_key
    dust_keys = [k for k in SED_params.keys() if "dust" in k]
    for k in dust_keys:
        print("...Saving {} to SED_params with values: {}".format(k, SED_params[k]))

    return SED_params
