"""
"""
# flake8: noqa

import subprocess


def compute_shasum(fn):
    command = "shasum {}".format(fn)
    raw_result = subprocess.check_output(command, shell=True)
    shasum = raw_result.strip().split()[0].decode()
    return shasum


from .load_diffsky_healpixel import (
    get_healpixel_bname_from_ra_dec_z,
    load_diffsky_params,
    load_healpixel,
)
