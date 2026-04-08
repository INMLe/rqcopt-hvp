import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from .util import get_matrices, get_tensors

def svd_retraction_single(X, W):
    u, s, vh = jnp.linalg.svd(X + W, full_matrices=False)
    return u @ vh

svd_retraction = jax.vmap(svd_retraction_single, in_axes=(0,0))

def qr_retraction_single(X, W):
    q, _ = jnp.linalg.qr(X + W)
    return q

qr_retraction = jax.vmap(qr_retraction_single, in_axes=(0,0))

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

@jax.jit
def batch_first_dim_Hilbert_Schmidt_inner_product(As_batched, Bs):
    return jax.vmap(
    Hilbert_Schmidt_inner_product, 
    in_axes=(0, None), # 0 for the list of tensors (As), None for the scalar max_bondim
    out_axes=0         # 0 for the output list of tensors (As)
)(As_batched, Bs)