"""
"""
# flake8: noqa

import typing

from diffmah.defaults import DEFAULT_MAH_PARAMS
from diffstar.defaults import DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS, FB
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY as DEFAULT_DSPS_COSMOLOGY
from jax import numpy as jnp

from .disk_bulge_modeling.disk_bulge_kernels import DEFAULT_FBULGE_PARAMS


class CosmoParams(typing.NamedTuple):
    """NamedTuple storing parameters of a flat w0-wa cdm cosmology"""

    Om0: jnp.float32
    w0: jnp.float32
    wa: jnp.float32
    h: jnp.float32
    fb: jnp.float32


DEFAULT_COSMO_PARAMS = CosmoParams(*DEFAULT_DSPS_COSMOLOGY, FB)
OUTER_RIM_COSMO_PARAMS = CosmoParams(0.2648, -1.0, 0.0, 0.71, 0.169)


DEFAULT_DIFFGAL_PARAMS = DEFAULT_MAH_PARAMS, DEFAULT_MS_PARAMS, DEFAULT_Q_PARAMS


ROMAN_RUBIN_2023_NSIDE = 32
ROMAN_RUBIN_2023_HPIX_PAT = "roman_rubin_2023_z_{0}_{1}_cutout_{2}.hdf5"
