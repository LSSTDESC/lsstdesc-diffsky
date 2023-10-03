"""
"""
import typing

from diffstar.defaults import FB
from dsps.cosmology.defaults import DEFAULT_COSMOLOGY as DEFAULT_DSPS_COSMOLOGY
from jax import numpy as jnp


class CosmoParams(typing.NamedTuple):
    """NamedTuple storing parameters of a flat w0-wa cdm cosmology"""

    Om0: jnp.float32
    w0: jnp.float32
    wa: jnp.float32
    h: jnp.float32
    fb: jnp.float32


DEFAULT_COSMO_PARAMS = CosmoParams(*DEFAULT_DSPS_COSMOLOGY, FB)
