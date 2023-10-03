"""
"""
import subprocess


def compute_shasum(fn):
    command = "shasum {}".format(fn)
    raw_result = subprocess.check_output(command, shell=True)
    shasum = raw_result.strip().split()[0].decode()
    return shasum
