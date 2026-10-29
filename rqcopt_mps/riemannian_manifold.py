import jax
import jax.numpy as jnp
from jax import config as c

from .util import get_matrices, get_tensors

c.update("jax_enable_x64", True)


def svd_retraction_single(X, W):
    u, s, vh = jnp.linalg.svd(X + W, full_matrices=False)
    return u @ vh


svd_retraction = jax.vmap(svd_retraction_single, in_axes=(0, 0))


@jax.jit
def retract_to_manifold(Gs, etas):
    Gs = get_matrices(Gs)
    etas = get_matrices(etas)
    res = svd_retraction(Gs, etas)
    return get_tensors(res)


@jax.jit
def project_to_tangent_space(Gs, Zs):
    """
    Project `z` onto the tangent plane at the unitary `u`.
    All inputs and outputs as tensors of shape (n,2,2,2,2).

    """
    gzg = jnp.einsum('eabmn, epqmn, epqcd -> eabcd', Gs, Zs.conj(), Gs)
    res = 0.5 * (Zs - gzg)
    return res


def inner_product_complex(As, Bs):
    return jnp.einsum('nijkl,nijkl->', As.conj(), Bs)


def Hilbert_Schmidt_inner_product(As, Bs):
    # If inner product is taken on tangent space it should be real-valued
    return jnp.einsum('nijkl,nijkl->', As.conj(), Bs).real


@jax.jit
def jitted_Hilbert_Schmidt_inner_product(As, Bs):
    return Hilbert_Schmidt_inner_product(As, Bs)
