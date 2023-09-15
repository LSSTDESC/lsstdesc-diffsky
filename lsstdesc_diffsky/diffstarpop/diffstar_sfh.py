"""
"""
from diffmah.individual_halo_assembly import calc_halo_history
from diffstar.fitting_helpers.stars import _integrate_sfr, _sfr_history_from_mah
from diffstar.sfh import get_sfh_from_mah_kern
from diffstar.utils import _jax_get_dt_array
from jax import jit as jjit
from jax import numpy as jnp
from jax import vmap

calculate_sfh_pop = jjit(vmap(_sfr_history_from_mah, in_axes=(*[None] * 2, *[0] * 4)))
integrate_sfr_pop = jjit(vmap(_integrate_sfr, in_axes=(0, None)))


def calculate_diffstar_sfh(
    tarr, t0, mah_params, ms_u_params, q_u_params, method="scan", sfh_min=1e-7
):
    lgt0 = jnp.log10(t0)
    dtarr = _jax_get_dt_array(tarr)
    lgtarr = jnp.log10(tarr)
    if method == "scan":
        sfh_kern = get_sfh_from_mah_kern(
            tobs_loop="scan", galpop_loop="vmap", lgt0=lgt0
        )
        sfh = sfh_kern(tarr, mah_params, ms_u_params, q_u_params)
        sfh = jnp.where(sfh <= sfh_min, sfh_min, sfh)
        smh = integrate_sfr_pop(sfh, dtarr)
    elif method == "vmap":
        mah_logmp = mah_params[:, 0]
        mah_lgtc = mah_params[:, 1]
        mah_early = mah_params[:, 2]
        mah_late = mah_params[:, 3]
        dmhdt, log_mah = calc_halo_history(
            tarr, t0, mah_logmp, 10**mah_lgtc, mah_early, mah_late
        )
        sfh = calculate_sfh_pop(lgtarr, dtarr, dmhdt, log_mah, ms_u_params, q_u_params)
        sfh = jnp.where(sfh <= sfh_min, sfh_min, sfh)
        smh = integrate_sfr_pop(sfh, dtarr)

    return smh, sfh
