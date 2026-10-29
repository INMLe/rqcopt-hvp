from functools import lru_cache, partial

import jax
import jax.numpy as jnp
from jax import config

from .brickwall_passes import (
    forward_pass_first_order, forward_pass_first_order_cached,
    backward_pass_first_order,
    forward_pass_second_order, backward_pass_second_order,
)
from .mps import compute_inner_product_from_mps
from .riemannian_manifold import (
    project_to_tangent_space, inner_product_complex,
)
from .util import batch_multiplication, transpose_tensors, adjoint_tensors

config.update("jax_enable_x64", True)


def _second_order_passes(
        psi0, phi0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw,
        flat_idxs_bw, layer_ends_bw, first_layer_increasing,
        left_canonical_init, truncation_dim, rel_tol, abs_tol, is_TI):
    '''
    Overlap, Frobenius gradient and HVP of a single sample.
    '''
    GTs, ZTs = transpose_tensors(Gs), transpose_tensors(Zs)
    psis, psi_Bs, _ = forward_pass_second_order(
        psi0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw,
        first_layer_increasing, truncation_dim, rel_tol, abs_tol, is_TI
    )
    overlap, grad_F, hvp_F, _ = backward_pass_second_order(
        phi0, psis, psi_Bs, GTs, ZTs, qubits, flat_idxs_bw, layer_ends_bw,
        left_canonical_init, truncation_dim, rel_tol, abs_tol, is_TI
    )
    return overlap, grad_F, hvp_F


#  -------------------------- FROBENIUS --------------------------


def _compute_loss_F_core(
        psi0, phi0, Gs, qubits, flat_idxs, layer_ends,
        first_layer_increasing, truncation_dim, rel_tol, abs_tol, is_TI=False
        ):
    ''' Jittable part. '''
    psi = forward_pass_first_order(
        psi0, Gs, qubits, flat_idxs, layer_ends,
        first_layer_increasing, truncation_dim, rel_tol, abs_tol, is_TI=is_TI
    )

    overlap = compute_inner_product_from_mps(phi0, psi)
    overlap_reduced = jax.lax.pmean(overlap, axis_name="d")
    # Same convention as compute_loss_gradient_F: C_F = 1 - Re<phi|psi>
    return 1. - overlap_reduced.real


def compute_loss_F(ket0, bra0, Gs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False):
    ''' layout: CircuitLayout from brickwall_circuit.get_circuit_layout '''
    core_func = _with_is_TI(_compute_loss_F_core, is_TI)

    # ------ Batching -----
    cost_F = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None),
        out_axes=0,
        axis_name="d",
        static_broadcasted_argnums=(6, 7, 8, 9),
    )(ket0, bra0, Gs, layout.qubits, layout.flat_idxs_fw, layout.layer_ends_fw,
      layout.first_layer_increasing, truncation_dim, rel_tol, abs_tol)
    # pmean inside the core already averages over the samples
    return cost_F[0]


def _compute_loss_gradient_F_core(
        psi0, phi0, Gs, qubits, flat_idxs_fw, layer_ends_fw,
        flat_idxs_bw, layer_ends_bw, first_layer_increasing,
        left_canonical_init, truncation_dim, rel_tol, abs_tol, is_TI=False
        ):
    ''' Jittable part. '''

    GTs = transpose_tensors(Gs)

    # -------- Forward pass --------
    psis = forward_pass_first_order_cached(
        psi0, Gs, qubits, flat_idxs_fw, layer_ends_fw,
        first_layer_increasing, truncation_dim, rel_tol, abs_tol, is_TI=is_TI
    )

    # -------- Backward pass --------
    overlap, grad_F = backward_pass_first_order(
        phi0=phi0, psis=psis, GTs=GTs, qubits=qubits, flat_idxs=flat_idxs_bw,
        layer_ends=layer_ends_bw, left_canonical_init=left_canonical_init,
        truncation_dim=truncation_dim, rel_tol=rel_tol, abs_tol=abs_tol,
        is_TI=is_TI
    )
    overlap_reduced = jax.lax.pmean(overlap, axis_name="d")
    grad_reduced = jax.lax.pmean(grad_F, axis_name="d")
    return 1. - overlap_reduced.real, grad_reduced


def compute_loss_gradient_F(ket0, bra0, Gs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False):
    core_func = _with_is_TI(_compute_loss_gradient_F_core, is_TI)

    # ------ Batching -----
    overlap, grad_F = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(0, 0),
        axis_name="d",
        static_broadcasted_argnums=(8, 9, 10, 11, 12),
    )(ket0, bra0, Gs, layout.qubits, layout.flat_idxs_fw, layout.layer_ends_fw,
      layout.flat_idxs_bw, layout.layer_ends_bw, layout.first_layer_increasing,
      layout.left_canonical_init, truncation_dim, rel_tol, abs_tol)

    return overlap[0], grad_F[0]


def _compute_loss_gradient_hvp_F_core(
        psi0, phi0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw,
        flat_idxs_bw, layer_ends_bw, first_layer_increasing,
        left_canonical_init, truncation_dim, rel_tol, abs_tol, is_TI=False):
    ''' Jittable part. '''
    overlap, grad_F_out, hvp_F_out = _second_order_passes(
        psi0, phi0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw,
        flat_idxs_bw, layer_ends_bw, first_layer_increasing,
        left_canonical_init, truncation_dim, rel_tol, abs_tol, is_TI
        )
    cost_reduced = jax.lax.pmean(overlap, axis_name="d")
    grad_reduced = jax.lax.pmean(grad_F_out, axis_name="d")
    hvp_reduced = jax.lax.pmean(hvp_F_out, axis_name="d")

    return 1.0 - cost_reduced.real, grad_reduced, hvp_reduced


def compute_loss_gradient_hvp_F(
        ket0, bra0, Gs, Zs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False):
    core_func = _with_is_TI(_compute_loss_gradient_hvp_F_core, is_TI)

    # ------ Batching -----
    cost, grad, hvp = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(0, 0, 0),
        axis_name="d",
        static_broadcasted_argnums=(9, 10, 11, 12, 13),
    )(ket0, bra0, Gs, Zs, layout.qubits, layout.flat_idxs_fw, layout.layer_ends_fw,
      layout.flat_idxs_bw, layout.layer_ends_bw, layout.first_layer_increasing,
      layout.left_canonical_init, truncation_dim, rel_tol, abs_tol)

    return cost[0], grad[0], hvp[0]


#  -------------------------- Hilbert-SChmidt test --------------------------

def _compute_loss_HST_core(
        psi0, phi0, Gs, qubits, flat_idxs, layer_ends,
        first_layer_increasing, truncation_dim, rel_tol, abs_tol, is_TI=False
        ):
    ''' Jittable part. '''

    psi = forward_pass_first_order(
        psi0, Gs, qubits, flat_idxs, layer_ends,
        first_layer_increasing, truncation_dim,
        rel_tol, abs_tol, is_TI=is_TI
    )
    overlap = compute_inner_product_from_mps(phi0, psi)
    fid = jnp.abs(overlap)**2
    fid_reduced = jax.lax.pmean(fid, axis_name="d")
    return 1.0 - fid_reduced


def compute_loss_HST(psi0, phi0, Gs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False):
    core_func = _with_is_TI(_compute_loss_HST_core, is_TI)

    # ------ Batching -----
    cost_HS = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None),
        out_axes=0,
        axis_name="d",
        static_broadcasted_argnums=(6, 7, 8, 9),
    )(psi0, phi0, Gs, layout.qubits, layout.flat_idxs_fw, layout.layer_ends_fw,
      layout.first_layer_increasing, truncation_dim, rel_tol, abs_tol)

    return cost_HS[0]


def _compute_loss_gradient_HST_core(
        psi0, phi0, Gs, qubits, flat_idxs_fw, layer_ends_fw,
        flat_idxs_bw, layer_ends_bw, first_layer_increasing,
        left_canonical_init, truncation_dim, rel_tol, abs_tol, is_TI=False
        ):

    GTs = transpose_tensors(Gs)

    psis = forward_pass_first_order_cached(
        psi0, Gs, qubits, flat_idxs_fw, layer_ends_fw,
        first_layer_increasing, truncation_dim, rel_tol, abs_tol, is_TI=is_TI
    )

    overlap, grad_F_out = backward_pass_first_order(
        phi0=phi0, psis=psis, GTs=GTs, qubits=qubits, flat_idxs=flat_idxs_bw,
        layer_ends=layer_ends_bw, left_canonical_init=left_canonical_init,
        truncation_dim=truncation_dim, rel_tol=rel_tol, abs_tol=abs_tol,
        is_TI=is_TI
    )

    cost_HS = jnp.abs(overlap)**2  # Compute cost function
    grad_HS = 2 * overlap*grad_F_out  # Compute T \nabla C_F for each summand

    cost_reduced = jax.lax.pmean(cost_HS, axis_name="d")
    grad_reduced = jax.lax.pmean(grad_HS, axis_name="d")
    return 1.0 - cost_reduced.real, grad_reduced


def compute_loss_gradient_HST(ket0, bra0, Gs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False):
    core_func = _with_is_TI(_compute_loss_gradient_HST_core, is_TI)

    # ------ Batching -----
    cost_HS, grad_HS = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(0, 0),
        axis_name="d",
        static_broadcasted_argnums=(8, 9, 10, 11, 12),
    )(ket0, bra0, Gs, layout.qubits, layout.flat_idxs_fw, layout.layer_ends_fw,
      layout.flat_idxs_bw, layout.layer_ends_bw, layout.first_layer_increasing,
      layout.left_canonical_init, truncation_dim, rel_tol, abs_tol)

    return cost_HS[0], grad_HS[0]


def compute_derivatives_from_F_to_HST(overlap, grad_F, hvp_F, Zs):
    ''' Convert derivatives of Frobenius norm to Hilbert-Schmidt test. '''

    Omega = -inner_product_complex(grad_F, Zs)

    cost_HS = jnp.abs(overlap)**2  # Compute cost function
    grad_HS = 2 * overlap * grad_F
    hvp_HS = 2 * (overlap * hvp_F + Omega * grad_F)

    return cost_HS, grad_HS, hvp_HS


def _compute_loss_gradient_hvp_HST_core(
        psi0, phi0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw, flat_idxs_bw,
        layer_ends_bw, first_layer_increasing, left_canonical_init, truncation_dim,
        rel_tol, abs_tol, is_TI=False
        ):
    ''' Jittable core of function. '''
    overlap, grad_F, hvp_F = _second_order_passes(
        psi0, phi0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw,
        flat_idxs_bw, layer_ends_bw, first_layer_increasing,
        left_canonical_init, truncation_dim, rel_tol, abs_tol, is_TI
        )

    # Consider HST
    cost_HS, grad_HS, hvp_HS = compute_derivatives_from_F_to_HST(
        overlap, grad_F, hvp_F, Zs)

    cost_reduced = jax.lax.pmean(cost_HS, axis_name="d")
    grad_reduced = jax.lax.pmean(grad_HS, axis_name="d")
    hvp_reduced = jax.lax.pmean(hvp_HS, axis_name="d")

    return 1.0 - cost_reduced.real, grad_reduced, hvp_reduced


def compute_loss_gradient_hvp_HST(
        psi0, phi0, Gs, Zs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False
        ):
    ''' Compute the derivatives for batched input states. '''
    core_func = _with_is_TI(_compute_loss_gradient_hvp_HST_core, is_TI)

    # ------ Batching -----
    cost, grad, hvp = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(0, 0, 0),
        axis_name="d",
        static_broadcasted_argnums=(9, 10, 11, 12, 13),
    )(psi0, phi0, Gs, Zs, layout.qubits, layout.flat_idxs_fw, layout.layer_ends_fw,
      layout.flat_idxs_bw, layout.layer_ends_bw, layout.first_layer_increasing,
      layout.left_canonical_init, truncation_dim, rel_tol, abs_tol)

    # Sum over samples
    return cost[0], grad[0], hvp[0]


#  -------------------------- RIEMANNIAN WRAPPER --------------------------

def compute_riemannian_loss_gradient_F(
        ket0, bra0, Gs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False
        ):
    # Euclidean quantities
    cost_F, grad_F = compute_loss_gradient_F(
        ket0, bra0, Gs, layout, truncation_dim, rel_tol, abs_tol, is_TI
        )

    # Project the gradient
    riem_grad_F = project_to_tangent_space(Gs, grad_F)

    return cost_F, riem_grad_F


def compute_riemannian_loss_gradient_HST(
        ket0, bra0, Gs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False
        ):
    ''' This function returns the cost and the Riemannian gradient. '''
    # Euclidean quantities
    cost_F, grad_F = compute_loss_gradient_HST(
        ket0, bra0, Gs, layout, truncation_dim, rel_tol, abs_tol, is_TI
        )

    # Project the gradient
    riem_grad_F = project_to_tangent_space(Gs, grad_F)

    return cost_F, riem_grad_F


def postprocess_euclidean_hvp(Gs, Zs, grad, hvp):
    ''' Project the Euclidean quantities to the Riemannian manifold. '''

    grad_dagger = adjoint_tensors(grad)

    # Project the HVP
    res = hvp - 0.5*(
        batch_multiplication(Zs, grad_dagger, Gs)
        + batch_multiplication(Gs, grad_dagger, Zs)
        )
    riem_hvp = project_to_tangent_space(Gs, res)
    return riem_hvp


def postprocess_euclidean_loss_gradient_hvp(Gs, Zs, grad, hvp):
    ''' Project the Euclidean quantities to the Riemannian manifold. '''

    grad_dagger = adjoint_tensors(grad)

    # Project the gradient
    riem_grad = project_to_tangent_space(Gs, grad)

    # Project the HVP
    res = hvp - 0.5*(
        batch_multiplication(Zs, grad_dagger, Gs)
        + batch_multiplication(Gs, grad_dagger, Zs)
        )
    riem_hvp = project_to_tangent_space(Gs, res)
    return riem_grad, riem_hvp


def compute_riemannian_loss_gradient_hvp_HST(
        ket0, bra0, Gs, Zs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False
        ):

    # Compute Euclidean derivatives
    cost, grad, hvp = compute_loss_gradient_hvp_HST(
        ket0, bra0, Gs, Zs, layout, truncation_dim, rel_tol, abs_tol, is_TI)

    # Project to Riemannian manifold
    riem_grad, riem_hvp = postprocess_euclidean_loss_gradient_hvp(Gs, Zs, grad, hvp)

    return cost, riem_grad, riem_hvp


def compute_riemannian_hvp_F(
        ket0, bra0, Gs, Zs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False):
    ''' This function only computes the HVP without loss and gradient. '''

    # Compute Euclidean derivatives
    cost, grad, hvp = compute_loss_gradient_hvp_F(
        ket0, bra0, Gs, Zs, layout, truncation_dim, rel_tol, abs_tol, is_TI
        )

    # Project to Riemannian manifold
    riem_hvp = postprocess_euclidean_hvp(Gs, Zs, grad, hvp)

    return riem_hvp


def compute_riemannian_hvp_HST(
        ket0, bra0, Gs, Zs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False):
    ''' This function only computes the HVP without loss and gradient. '''

    # Compute Euclidean derivatives
    cost, grad, hvp = compute_loss_gradient_hvp_HST(
        ket0, bra0, Gs, Zs, layout, truncation_dim, rel_tol, abs_tol, is_TI
        )

    # Project to Riemannian manifold
    riem_hvp = postprocess_euclidean_hvp(Gs, Zs, grad, hvp)

    return riem_hvp


def compute_loss_HST_loop(psis, phis, Gs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False):
    ''' Compute the loss for n_samples > n_cpus '''
    N = psis.shape[0]
    num_devices = jax.local_device_count()

    # --- Padding and reshaping ---
    remainder = N % num_devices
    if remainder != 0:
        raise ValueError(
            f"Number of samples ({N}) must be divisible by number of devices ({num_devices}). "
            f"Got remainder: {remainder}. Please adjust the batch size or pad the input."
        )

    num_chunks = psis.shape[0] // num_devices
    batched_psis = psis.reshape(num_devices, num_chunks, *psis.shape[1:])
    batched_phis = phis.reshape(num_devices, num_chunks, *phis.shape[1:])

    core_func = _with_is_TI(_compute_loss_HST_core, is_TI)
    pmapped_fn = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None),
        out_axes=0,
        axis_name="d",
        static_broadcasted_argnums=(6, 7, 8, 9),
        batched=True
    )
    fid = pmapped_fn(
            batched_psis, batched_phis, Gs, layout.qubits,
            layout.flat_idxs_fw, layout.layer_ends_fw, layout.first_layer_increasing,
            truncation_dim, rel_tol, abs_tol
        )
    # reduce over all axes
    return jnp.mean(fid)


def compute_riemannian_loss_gradient_HST_loop(
        psis, phis, Gs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False
        ):
    ''' Compute the loss and Riemannian gradient for n_samples > n_cpus '''
    N = psis.shape[0]
    num_devices = jax.local_device_count()

    # --- Padding and reshaping ---
    remainder = N % num_devices
    if remainder != 0:
        raise ValueError(
            f"Number of samples ({N}) must be divisible by number of devices ({num_devices}). "
            f"Got remainder: {remainder}. Please adjust the batch size or pad the input."
        )

    num_chunks = psis.shape[0] // num_devices
    batched_psis = psis.reshape(num_devices, num_chunks, *psis.shape[1:])
    batched_phis = phis.reshape(num_devices, num_chunks, *phis.shape[1:])

    core_func = _with_is_TI(_compute_loss_gradient_HST_core, is_TI)

    pmapped_fn = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(0, 0),
        axis_name="d",
        static_broadcasted_argnums=(8, 9, 10, 11, 12),
        batched=True
    )

    cost_HS, grad_HS = pmapped_fn(
        batched_psis, batched_phis, Gs, layout.qubits,
        layout.flat_idxs_fw, layout.layer_ends_fw, layout.flat_idxs_bw, layout.layer_ends_bw,
        layout.first_layer_increasing, layout.left_canonical_init, truncation_dim,
        rel_tol, abs_tol
    )

    # Average over all samples
    cost_mean = jnp.mean(cost_HS)
    grad_mean = jnp.mean(grad_HS, axis=(0, 1))

    # Project gradient to Riemannian manifold
    riem_grad = project_to_tangent_space(Gs, grad_mean)

    return cost_mean, riem_grad


def compute_riemannian_loss_gradient_hvp_HST_loop(
        psis, phis, Gs, Zs, layout, truncation_dim, rel_tol, abs_tol, is_TI=False
        ):
    ''' Compute the loss, Riemannian gradient, and HVP for n_samples > n_cpus '''
    N = psis.shape[0]
    num_devices = jax.local_device_count()

    # --- Padding and reshaping ---
    remainder = N % num_devices
    if remainder != 0:
        raise ValueError(
            f"Number of samples ({N}) must be divisible by number of devices ({num_devices}). "
            f"Got remainder: {remainder}. Please adjust the batch size or pad the input."
        )

    num_chunks = psis.shape[0] // num_devices
    batched_psis = psis.reshape(num_devices, num_chunks, *psis.shape[1:])
    batched_phis = phis.reshape(num_devices, num_chunks, *phis.shape[1:])

    core_func = _with_is_TI(_compute_loss_gradient_hvp_HST_core, is_TI)

    pmapped_fn = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(0, 0, 0),
        axis_name="d",
        static_broadcasted_argnums=(9, 10, 11, 12, 13),
        batched=True
    )

    cost_HS, grad_HS, hvp_HS = pmapped_fn(
        batched_psis, batched_phis, Gs, Zs, layout.qubits,
        layout.flat_idxs_fw, layout.layer_ends_fw, layout.flat_idxs_bw, layout.layer_ends_bw,
        layout.first_layer_increasing, layout.left_canonical_init, truncation_dim,
        rel_tol, abs_tol
    )

    # Average over all samples
    cost_mean = jnp.mean(cost_HS)
    grad_mean = jnp.mean(grad_HS, axis=(0, 1))
    hvp_mean = jnp.mean(hvp_HS, axis=(0, 1))

    # Project to Riemannian manifold
    riem_grad, riem_hvp = postprocess_euclidean_loss_gradient_hvp(Gs, Zs, grad_mean, hvp_mean)

    return cost_mean, riem_grad, riem_hvp


@lru_cache(maxsize=None)
def _with_is_TI(core_func, is_TI):
    '''
    core_func with is_TI fixed. Cached so the same object is returned for the same
    arguments, which keeps the cache of pmap_maybe_batched (and the compilation) working.
    '''
    return partial(core_func, is_TI=is_TI)


@lru_cache(maxsize=None)
def pmap_maybe_batched(f, in_axes, out_axes, axis_name, static_broadcasted_argnums, batched=False):
    '''
    Parallelize f over the devices (and, if batched, vectorize over the samples per device).
    The wrapped function is cached: building a new jax.vmap/jax.pmap on every call would make
    JAX trace and compile f again each time. All arguments must be hashable (functions, tuples).
    '''
    if batched:
        # After pmap consumes axis 0, vmap over the remaining axis 0 (chunks)
        # So vmap_in_axes is the same as in_axes
        vmapped_f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)

        # Then pmap over devices (outer)
        return jax.pmap(
            vmapped_f,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            static_broadcasted_argnums=static_broadcasted_argnums
        )
    else:
        return jax.pmap(
            f,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            static_broadcasted_argnums=static_broadcasted_argnums
        )
