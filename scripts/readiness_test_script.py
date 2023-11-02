"""Run basic diagnostics on existence of NaNs and min/max range of each data column
"""
import argparse
import os
from glob import glob

import numpy as np
from lsstdesc_diffsky.io_utils.load_diffsky_healpixel import load_healpixel

HPIX_BNAME_PAT = "roman_rubin_2023_*.hdf5"
LCRC_DRN = "/lcrc/project/galsampler/Catalog_5000/OR_5000/diffsky_v1.0.1/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-drn", help="input directory", default=LCRC_DRN)
    parser.add_argument("-bpat", help="basename pattern", default=HPIX_BNAME_PAT)
    args = parser.parse_args()

    bpat = args.bpat
    drn = args.drn

    fname_list = glob(os.path.join(drn, bpat))
    drnout = os.path.join(drn, "readiness_tests")
    os.makedirs(drnout, exist_ok=True)

    problem_dict = dict()
    for fn in fname_list:
        mock, metadata = load_healpixel(fn)
        keylist = sorted(list(mock.keys()))
        bname = os.path.basename(fn)
        bnout = bname.replace(".hdf5", ".readiness.txt")
        fnout = os.path.join(drnout, bnout)
        print("...running readiness test on {0}".format(bname))

        nan_collector = []
        with open(fnout, "w") as fout:
            fout.write("# colname  all-finite  min  max")
            for key in keylist:
                xmin = np.nanmin(mock[key])
                xmax = np.nanmax(mock[key])
                all_finite = np.all(np.isfinite(mock[key]))

                line_out_pat = "{0}  {1}  {2:.2f}  {3:.2f}\n"
                line_out = line_out_pat.format(key, all_finite, xmin, xmax)
                fout.write(line_out)
                if all_finite is False:
                    nan_collector.append(key)

        if len(nan_collector) > 0:
            msg = "\nFor {0} the following columns contained NaNs:"
            print(msg.format(bname))
            print(nan_collector)

            problem_dict[bname] = nan_collector

    fnout_summary = os.path.join(drnout, "readiness_summary.txt")
    with open(fnout_summary, "w") as fout:
        fout.write("#  fname  nancols\n")
        for bname, nancols in problem_dict.items():
            line_out = bname + " " + " ".join(nancols) + "\n"
            fout.write(line_out)
