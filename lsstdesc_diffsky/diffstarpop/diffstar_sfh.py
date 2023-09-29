"""
"""
from diffstar.defaults import FB
from diffstar.fitting_helpers.fitting_kernels import _integrate_sfr
from diffstar.sfh import get_sfh_from_mah_kern
from diffstar.utils import _jax_get_dt_array
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

integrate_sfr_pop = jjit(vmap(_integrate_sfr, in_axes=(0, None)))


def calculate_diffstar_sfh(
    tarr, t0, mah_params, ms_u_params, q_u_params, method="scan", sfh_min=1e-7
):
    lgt0 = jnp.log10(t0)
    dtarr = _jax_get_dt_array(tarr)
    if method == "scan":
        sfh_kern = get_sfh_from_mah_kern(tobs_loop="scan", galpop_loop="vmap")
        sfh = sfh_kern(tarr, mah_params, ms_u_params, q_u_params, lgt0, FB)
        sfh = jnp.where(sfh <= sfh_min, sfh_min, sfh)
        smh = integrate_sfr_pop(sfh, dtarr)
    elif method == "vmap":
        mah_logmp = mah_params[:, 0]
        mah_lgtc = mah_params[:, 1]
        mah_early = mah_params[:, 2]
        mah_late = mah_params[:, 3]
        mah_params = jnp.array((mah_logmp, mah_lgtc, mah_early, mah_late)).T
        sfh_kern = get_sfh_from_mah_kern(tobs_loop="vmap", galpop_loop="vmap")
        sfh = sfh_kern(tarr, mah_params, ms_u_params, q_u_params, lgt0, FB)
        sfh = jnp.where(sfh <= sfh_min, sfh_min, sfh)
        smh = integrate_sfr_pop(sfh, dtarr)

    return smh, sfh
