import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from .util import (
    batch_multiplication, 
    transpose_tensors, 
    flatten, 
    adjoint_tensors
)
from .brickwall_circuit import (
    map_list_of_gates_to_layers, 
    gate_is_at_boundary
)
from .mps import compute_inner_product_from_mps
from .riemannian_manifold import (
    project_to_tangent_space, 
    inner_product_complex
)
from .brickwall_passes import (
    forward_pass_first_order, 
    forward_pass_first_order_TI,
    forward_pass_first_order_cached, 
    forward_pass_first_order_cached_TI,
    forward_pass_second_order, 
    forward_pass_second_order_TI, 
    backward_pass_first_order, 
    backward_pass_first_order_TI,
    backward_pass_second_order, 
    backward_pass_second_order_TI
)

#  ############################# FROBENIUS #############################

def _compute_loss_F_core(
        psi0, phi0, Gs, qubits, flat_idxs, layer_ends, 
        first_layer_increasing, truncation_dim
        ):
    ''' Jittable part. '''
    psi = forward_pass_first_order(
        psi0, Gs, qubits, flat_idxs, layer_ends,
        first_layer_increasing, truncation_dim
    )

    overlap = compute_inner_product_from_mps(phi0, psi)
    overlap_reduced = jax.lax.pmean(overlap, axis_name="d")
    return overlap_reduced.real

def _compute_loss_F_core_TI(
        psi0, phi0, Gs, qubits, flat_idxs, layer_ends, 
        first_layer_increasing, truncation_dim
        ):
    ''' Jittable part. '''
    psi = forward_pass_first_order_TI(
        psi0, Gs, qubits, flat_idxs, layer_ends,
        first_layer_increasing, truncation_dim
    )

    overlap = compute_inner_product_from_mps(phi0, psi)
    overlap_reduced = jax.lax.pmean(overlap, axis_name="d")
    return overlap_reduced.real

def compute_loss_F(ket0, bra0, Gs, qubits, first_layer_increasing, truncation_dim, is_TI=False):
    # ------ Non-jittable preprocessing ------
    idx_per_layer = map_list_of_gates_to_layers(qubits)
    flat_idxs = flatten(idx_per_layer)
    layer_ends = gate_is_at_boundary(idx_per_layer)

    if is_TI: core_func = _compute_loss_F_core_TI
    else: core_func = _compute_loss_F_core

    # ------ Batching -----
    cost_F = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None),
        out_axes=0,
        axis_name="d",
        static_broadcasted_argnums=(6, 7),
    )(ket0, bra0, Gs, qubits, flat_idxs, layer_ends, 
      first_layer_increasing, truncation_dim)
    #TODO: Before, this was a sum, but it should be a mean over the samples?
    return cost_F[0]

def _compute_loss_gradient_F_core(
        psi0, phi0, Gs, qubits, flat_idxs_fw, layer_ends_fw, 
        flat_idxs_bw, layer_ends_bw, first_layer_increasing, 
        left_canonical_init, truncation_dim, rel_tol, abs_tol
        ):
    ''' Jittable part. '''

    GTs = transpose_tensors(Gs)

    # -------- Forward pass --------
    psis = forward_pass_first_order_cached(
        psi0, Gs, qubits, flat_idxs_fw, layer_ends_fw,
        first_layer_increasing, truncation_dim, rel_tol, abs_tol
    )

    # -------- Backward pass --------
    overlap, grad_F = backward_pass_first_order(
        phi0=phi0, psis=psis, GTs=GTs, qubits=qubits, flat_idxs=flat_idxs_bw,
        layer_ends=layer_ends_bw, left_canonical_init=left_canonical_init,
        truncation_dim=truncation_dim, rel_tol=rel_tol, abs_tol=abs_tol
    )
    overlap_reduced = jax.lax.pmean(overlap, axis_name="d")
    grad_reduced = jax.lax.pmean(grad_F, axis_name="d")
    return 1. - overlap_reduced.real, grad_reduced

def _compute_loss_gradient_F_core_TI(
        psi0, phi0, Gs, qubits, flat_idxs_fw, layer_ends_fw, 
        flat_idxs_bw, layer_ends_bw, first_layer_increasing, 
        left_canonical_init, truncation_dim, rel_tol, abs_tol
        ):
    ''' Jittable part. '''

    GTs = transpose_tensors(Gs)

    # -------- Forward pass --------
    psis = forward_pass_first_order_cached_TI(
        psi0, Gs, qubits, flat_idxs_fw, layer_ends_fw,
        first_layer_increasing, truncation_dim, rel_tol, abs_tol
    )

    # -------- Backward pass --------
    overlap, grad_F = backward_pass_first_order_TI(
        phi0=phi0, psis=psis, GTs=GTs, qubits=qubits, flat_idxs=flat_idxs_bw,
        layer_ends=layer_ends_bw, left_canonical_init=left_canonical_init,
        truncation_dim=truncation_dim, rel_tol=rel_tol, abs_tol=abs_tol
    )
    overlap_reduced = jax.lax.pmean(overlap, axis_name="d")
    grad_reduced = jax.lax.pmean(grad_F, axis_name="d")
    return 1. - overlap_reduced.real, grad_reduced

def compute_loss_gradient_F(ket0, bra0, Gs, qubits, first_layer_increasing, 
                            truncation_dim, rel_tol, abs_tol, is_TI=False):

    # ---- Forward pass ------
    idx_per_layer_fw = map_list_of_gates_to_layers(qubits)  # indices for gates 2...n
    flat_idxs_fw = flatten(idx_per_layer_fw)
    layer_ends_fw = gate_is_at_boundary(idx_per_layer_fw)

    # ---- Backward pass -----
    idx_per_layer_bw = idx_per_layer_fw[::-1]  # reversed layer order
    flat_idxs_bw = flat_idxs_fw[::-1]
    layer_ends_bw = gate_is_at_boundary(idx_per_layer_bw)
    
    nlayer_is_even = bool(len(idx_per_layer_bw[0])!=0)
    left_canonical_init = (nlayer_is_even == first_layer_increasing)

    if is_TI: core_func = _compute_loss_gradient_F_core_TI
    else: core_func = _compute_loss_gradient_F_core
    
    # ------ Batching -----
    overlap, grad_F = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(0, 0),
        axis_name="d",
        static_broadcasted_argnums=(9, 10, 11, 12),
    )(ket0, bra0, Gs, qubits, flat_idxs_fw, layer_ends_fw, flat_idxs_bw, 
      layer_ends_bw, first_layer_increasing, left_canonical_init, truncation_dim, rel_tol, abs_tol)

    return overlap[0], grad_F[0]

def _compute_loss_gradient_hvp_F_core(
        psi0, phi0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw, 
        flat_idxs_bw, layer_ends_bw, first_layer_increasing, 
        left_canonical_init, truncation_dim, rel_tol, abs_tol):
    ''' Jittable part. '''

    GTs, ZTs = transpose_tensors(Gs), transpose_tensors(Zs)
 
    # -------- Forward pass --------
    psis, Dpsis = forward_pass_second_order(
        psi0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw,
        first_layer_increasing, truncation_dim, rel_tol, abs_tol
    )

    # -------- Backward pass --------
    overlap, grad_F_out, hvp_F_out = backward_pass_second_order(
        phi0, psis, Dpsis, GTs, ZTs, qubits, flat_idxs_bw,
        layer_ends_bw, left_canonical_init, truncation_dim,
        rel_tol, abs_tol
        )
    cost_reduced = jax.lax.pmean(overlap, axis_name="d")
    grad_reduced = jax.lax.pmean(grad_F_out, axis_name="d")
    hvp_reduced = jax.lax.pmean(hvp_F_out, axis_name="d")

    return 1.0 - cost_reduced.real, grad_reduced, hvp_reduced

def _compute_loss_gradient_hvp_F_core_TI(
    psi0, phi0,
    Gs, Zs,
    qubits,
    flat_idxs_fw, layer_ends_fw,
    flat_idxs_bw, layer_ends_bw,
    first_layer_increasing,
    left_canonical_init,
    truncation_dim,
    rel_tol, abs_tol
):
    """Jittable part (TI-per-layer). Same structure as original."""

    GTs, ZTs = transpose_tensors(Gs), transpose_tensors(Zs)

    # -------- Forward pass --------
    psis, Dpsis = forward_pass_second_order_TI(
        psi0, Gs, Zs, qubits,
        flat_idxs_fw, layer_ends_fw,
        first_layer_increasing, truncation_dim,
        rel_tol, abs_tol
    )

    # -------- Backward pass --------
    overlap, grad_F_out, hvp_F_out = backward_pass_second_order_TI(
        phi0, psis, Dpsis, GTs, ZTs, qubits,
        flat_idxs_bw, layer_ends_bw,
        left_canonical_init, truncation_dim, 
        rel_tol, abs_tol
    )
    cost_reduced = jax.lax.pmean(overlap, axis_name="d")
    grad_reduced = jax.lax.pmean(grad_F_out, axis_name="d")
    hvp_reduced = jax.lax.pmean(hvp_F_out, axis_name="d")

    return 1.0 - cost_reduced.real, grad_reduced, hvp_reduced

def compute_loss_gradient_hvp_F(
        ket0, bra0, Gs, Zs, qubits, first_layer_increasing, 
        truncation_dim, rel_tol, abs_tol, is_TI=False):
        
    # -------- Forward pass --------
    idx_per_layer_fw = map_list_of_gates_to_layers(qubits)
    flat_idxs_fw = flatten(idx_per_layer_fw)
    layer_ends_fw = gate_is_at_boundary(idx_per_layer_fw)

    # -------- Backward pass --------
    idx_per_layer_bw = idx_per_layer_fw[::-1]
    flat_idxs_bw = flat_idxs_fw[::-1]
    layer_ends_bw = gate_is_at_boundary(idx_per_layer_bw)

    even_nlayers = bool(len(idx_per_layer_bw[0])!=0)  # Odd layers a filled with empty list
    left_canonical_init = (even_nlayers == first_layer_increasing)

    if is_TI: core_func = _compute_loss_gradient_hvp_F_core_TI
    else: core_func = _compute_loss_gradient_hvp_F_core

    # ------ Batching -----
    cost, grad, hvp = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(0, 0, 0),
        axis_name="d",
        static_broadcasted_argnums=(9, 10, 11, 12, 13),
    )(ket0, bra0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw, flat_idxs_bw, 
      layer_ends_bw, first_layer_increasing, left_canonical_init, truncation_dim, rel_tol, abs_tol)
    
    return cost[0], grad[0], hvp[0]


#  ########################## Hilbert-SChmidt test ##########################

def _compute_loss_HST_core(
        psi0, phi0, Gs, qubits, flat_idxs, layer_ends, 
        first_layer_increasing, truncation_dim, rel_tol, abs_tol
        ):
    ''' Jittable part. '''
    
    psi = forward_pass_first_order(
        psi0, Gs, qubits, flat_idxs, layer_ends,
        first_layer_increasing, truncation_dim,
        rel_tol, abs_tol
    )
    overlap = compute_inner_product_from_mps(phi0, psi)
    fid = jnp.abs(overlap)**2
    fid_reduced = jax.lax.pmean(fid, axis_name="d")
    return 1.0 - fid_reduced

def _compute_loss_HST_core_TI(
        psi0, phi0, Gs, qubits, flat_idxs, layer_ends, 
        first_layer_increasing, truncation_dim, rel_tol, abs_tol
        ):
    ''' Jittable part. '''

    psi = forward_pass_first_order_TI(
        psi0, Gs, qubits, flat_idxs, layer_ends,
        first_layer_increasing, truncation_dim,
        rel_tol, abs_tol
    )
    overlap = compute_inner_product_from_mps(phi0, psi)
    fid = jnp.abs(overlap)**2
    fid_reduced = jax.lax.pmean(fid, axis_name="d")
    return 1.0 - fid_reduced

def compute_loss_HST(psi0, phi0, Gs, qubits, first_layer_increasing,  
                     truncation_dim, rel_tol, abs_tol, is_TI=False):

    # ------ Non-jittable preprocessing ------
    idx_per_layer = map_list_of_gates_to_layers(qubits)  # indices for gates 2...n
    flat_idxs = flatten(idx_per_layer)
    layer_ends = gate_is_at_boundary(idx_per_layer)

    if is_TI: core_func = _compute_loss_HST_core_TI
    else: core_func = _compute_loss_HST_core

    # ------ Batching -----
    cost_HS = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None),
        out_axes=0,
        axis_name="d",
        static_broadcasted_argnums=(6, 7, 8, 9),
    )(psi0, phi0, Gs, qubits, flat_idxs, layer_ends, first_layer_increasing, truncation_dim, rel_tol, abs_tol)
    
    return cost_HS[0]

def _compute_loss_gradient_HST_core(
        psi0, phi0, Gs, qubits, flat_idxs_fw, layer_ends_fw, 
        flat_idxs_bw, layer_ends_bw, first_layer_increasing, 
        left_canonical_init, truncation_dim, rel_tol, abs_tol
        ):
    
    GTs = transpose_tensors(Gs)

    psis = forward_pass_first_order_cached(
        psi0, Gs, qubits, flat_idxs_fw, layer_ends_fw,
        first_layer_increasing, truncation_dim, rel_tol, abs_tol
    )

    overlap, grad_F_out = backward_pass_first_order(
        phi0=phi0, psis=psis, GTs=GTs, qubits=qubits, flat_idxs=flat_idxs_bw,
        layer_ends=layer_ends_bw, left_canonical_init=left_canonical_init,
        truncation_dim=truncation_dim, rel_tol=rel_tol, abs_tol=abs_tol
    )

    cost_HS = jnp.abs(overlap)**2  # Compute cost function
    grad_HS = 2 * overlap*grad_F_out  # Compute T \nabla C_F for each summand
   
    cost_reduced = jax.lax.pmean(cost_HS, axis_name="d")
    grad_reduced = jax.lax.pmean(grad_HS, axis_name="d")
    return 1.0 - cost_reduced.real, grad_reduced

def _compute_loss_gradient_HST_core_TI(
        psi0, phi0, Gs, qubits, flat_idxs_fw, layer_ends_fw, 
        flat_idxs_bw, layer_ends_bw, first_layer_increasing, 
        left_canonical_init, truncation_dim, rel_tol, abs_tol
        ):
    
    GTs = transpose_tensors(Gs)

    psis = forward_pass_first_order_cached_TI(
        psi0, Gs, qubits, flat_idxs_fw, layer_ends_fw,
        first_layer_increasing, truncation_dim, rel_tol, abs_tol
    )

    overlap, grad_F_out = backward_pass_first_order_TI(
        phi0=phi0, psis=psis, GTs=GTs, qubits=qubits, flat_idxs=flat_idxs_bw,
        layer_ends=layer_ends_bw, left_canonical_init=left_canonical_init,
        truncation_dim=truncation_dim, rel_tol=rel_tol, abs_tol=abs_tol
    )

    cost_HS = jnp.abs(overlap)**2  # Compute cost function
    grad_HS = 2 * overlap*grad_F_out  # Compute T \nabla C_F for each summand
    cost_reduced = jax.lax.pmean(cost_HS, axis_name="d")
    grad_reduced = jax.lax.pmean(grad_HS, axis_name="d")
    return 1.0 - cost_reduced.real, grad_reduced

def compute_loss_gradient_HST(ket0, bra0, Gs, qubits, first_layer_increasing, 
                              truncation_dim, rel_tol, abs_tol, is_TI=False):
    # ----- Forward pass -----
    idx_per_layer_fw = map_list_of_gates_to_layers(qubits)  # indices for gates 2..n
    flat_idxs_fw = flatten(idx_per_layer_fw)
    layer_ends_fw = gate_is_at_boundary(idx_per_layer_fw)

    # ----- Backward pass -----
    idx_per_layer_bw = idx_per_layer_fw[::-1]  # reversed layer order
    flat_idxs_bw = flat_idxs_fw[::-1]
    layer_ends_bw = gate_is_at_boundary(idx_per_layer_bw)

    nlayer_is_even = bool(len(idx_per_layer_bw[0]))
    left_canonical_init = (nlayer_is_even == first_layer_increasing)

    if is_TI: core_func = _compute_loss_gradient_HST_core_TI
    else: core_func = _compute_loss_gradient_HST_core

    # ------ Batching -----
    cost_HS, grad_HS = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(0, 0),
        axis_name="d",
        static_broadcasted_argnums=(8, 9, 10, 11, 12),
    )(ket0, bra0, Gs, qubits, flat_idxs_fw, layer_ends_fw, 
      flat_idxs_bw, layer_ends_bw, first_layer_increasing, 
      left_canonical_init, truncation_dim, rel_tol, abs_tol)

    return cost_HS[0], grad_HS[0]
    

def compute_derivatives_from_F_to_HST(overlap, grad_F, hvp_F, Zs):
    ''' Convert derivatives of Frobenius norm to Hilbert-Schmidt test. '''

    Omega = -inner_product_complex(grad_F, Zs)

    cost_HS = jnp.abs(overlap)**2  # Compute cost function   
    grad_HS = 2 * overlap * grad_F
    hvp_HS = 2 * (overlap * hvp_F + Omega * grad_F)

    return cost_HS, grad_HS, hvp_HS

def compute_hvp_from_F_to_HST(overlap, grad_F, hvp_F, Zs):
    ''' Convert derivatives of Frobenius norm to Hilbert-Schmidt test. '''

    Omega = -inner_product_complex(grad_F, Zs)
    hvp_HS = 2 * (overlap * hvp_F + Omega * grad_F)

    return hvp_HS

def _compute_loss_gradient_hvp_HST_core(
        psi0, phi0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw, flat_idxs_bw,
        layer_ends_bw, first_layer_increasing, left_canonical_init, truncation_dim,
        rel_tol, abs_tol
        ):
    ''' Jittable core of function. '''

    GTs, ZTs = transpose_tensors(Gs), transpose_tensors(Zs)

    # ----- Forward pass -----
    psis, Dpsis = forward_pass_second_order(
        psi0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw, 
        first_layer_increasing, truncation_dim, rel_tol, abs_tol
        )
    
    # ----- Backward pass -----
    overlap, grad_F, hvp_F = backward_pass_second_order(
        phi0, psis, Dpsis, GTs, ZTs, qubits, flat_idxs_bw, 
        layer_ends_bw, left_canonical_init, truncation_dim,
        rel_tol, abs_tol
        )

    # Consider HST
    cost_HS, grad_HS, hvp_HS = compute_derivatives_from_F_to_HST(
        overlap, grad_F, hvp_F, Zs)
    
    cost_reduced = jax.lax.pmean(cost_HS, axis_name="d")
    grad_reduced = jax.lax.pmean(grad_HS, axis_name="d")
    hvp_reduced = jax.lax.pmean(hvp_HS, axis_name="d")

    return 1.0 - cost_reduced.real, grad_reduced, hvp_reduced

def _compute_loss_gradient_hvp_HST_core_TI(
        psi0, phi0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw, flat_idxs_bw,
        layer_ends_bw, first_layer_increasing, left_canonical_init, truncation_dim,
        rel_tol, abs_tol
        ):
    ''' Jittable core of function. '''

    GTs, ZTs = transpose_tensors(Gs), transpose_tensors(Zs)

    # ----- Forward pass -----
    psis, Dpsis = forward_pass_second_order_TI(
        psi0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw, 
        first_layer_increasing, truncation_dim, rel_tol, abs_tol
        )
    
    # ----- Backward pass -----
    overlap, grad_F, hvp_F = backward_pass_second_order_TI(
        phi0, psis, Dpsis, GTs, ZTs, qubits, flat_idxs_bw, 
        layer_ends_bw, left_canonical_init, truncation_dim,
        rel_tol, abs_tol
        )

    # Consider HST
    cost_HS, grad_HS, hvp_HS = compute_derivatives_from_F_to_HST(
        overlap, grad_F, hvp_F, Zs)
    # Reduce within pmap across the device axis
    cost_reduced = jax.lax.pmean(cost_HS, axis_name="d")
    grad_reduced = jax.lax.pmean(grad_HS, axis_name="d")
    hvp_reduced = jax.lax.pmean(hvp_HS, axis_name="d")
    return 1.0 - cost_reduced.real, grad_reduced, hvp_reduced

def compute_loss_gradient_hvp_HST(
        psi0, phi0, Gs, Zs, qubits, first_layer_increasing, 
        truncation_dim, rel_tol, abs_tol, is_TI=False
        ):
    ''' Compute the derivatives for batched input states. '''

    # ------- Forward --------
    idx_per_layer_fw = map_list_of_gates_to_layers(qubits)
    flat_idxs_fw = flatten(idx_per_layer_fw)
    layer_ends_fw = gate_is_at_boundary(idx_per_layer_fw)

    # ------- Backward --------
    idx_per_layer_bw = idx_per_layer_fw[::-1]
    flat_idxs_bw = flat_idxs_fw[::-1]
    layer_ends_bw = gate_is_at_boundary(idx_per_layer_bw)

    # ------- Determine initial direction of sweeping -------
    # Odd nlayers are filled with zero layer
    even_nlayers = bool(len(idx_per_layer_bw[0])!=0)  
    left_canonical_init = (even_nlayers == first_layer_increasing)

    if is_TI: core_func = _compute_loss_gradient_hvp_HST_core_TI
    else: core_func = _compute_loss_gradient_hvp_HST_core

    # ------ Batching -----
    cost, grad, hvp = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(0, 0, 0),
        axis_name="d",
        static_broadcasted_argnums=(9, 10, 11, 12, 13),
    )(psi0, phi0, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw, flat_idxs_bw, 
      layer_ends_bw, first_layer_increasing, left_canonical_init, truncation_dim, rel_tol, abs_tol)

    # Sum over samples
    return cost[0], grad[0], hvp[0]

    
#  ############################# RIEMANNIAN WRAPPER #############################

def compute_riemannian_loss_gradient_F(ket0, bra0, Gs, qubits, first_layer_increasing, truncation_dim=128, is_TI=False):
    # Euclidean quantities
    cost_F, grad_F = compute_loss_gradient_F(ket0, bra0, Gs, qubits, first_layer_increasing, truncation_dim, is_TI)

    # Project the gradient
    riem_grad_F = project_to_tangent_space(Gs, grad_F)

    return cost_F, riem_grad_F

def compute_riemannian_loss_gradient_HST(
        ket0, bra0, Gs, qubits, first_layer_increasing, 
        truncation_dim, rel_tol, abs_tol, is_TI=False
        ):
    ''' This function returns the cost and the Riemannian gradient. '''
    # Euclidean quantities
    cost_F, grad_F = compute_loss_gradient_HST(
        ket0, bra0, Gs, qubits, first_layer_increasing, 
        truncation_dim, rel_tol, abs_tol, is_TI
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

def compute_riemannian_loss_gradient_hvp_F(
        ket0, bra0, Gs, Zs, qubits, first_layer_increasing=True, truncation_dim=128, is_TI=False):
    
    # Compute Euclidean derivatives
    cost, grad, hvp = compute_loss_gradient_hvp_F(
            ket0, bra0, Gs, Zs, qubits, first_layer_increasing, truncation_dim, is_TI)
        
    # Project to Riemannian manifold
    riem_grad, riem_hvp = postprocess_euclidean_loss_gradient_hvp(Gs, Zs, grad, hvp)

    return cost, riem_grad, riem_hvp

def compute_riemannian_loss_gradient_hvp_HST(
        ket0, bra0, Gs, Zs, qubits, first_layer_increasing, 
        truncation_dim, rel_tol, abs_tol, is_TI=False
        ):
    
    # Compute Euclidean derivatives
    cost, grad, hvp = compute_loss_gradient_hvp_HST(
        ket0, bra0, Gs, Zs, qubits, first_layer_increasing, truncation_dim, rel_tol, abs_tol, is_TI)
   
   # Project to Riemannian manifold
    riem_grad, riem_hvp = postprocess_euclidean_loss_gradient_hvp(Gs, Zs, grad, hvp)

    return cost, riem_grad, riem_hvp

def compute_riemannian_hvp_F(
        ket0, bra0, Gs, Zs, qubits, first_layer_increasing, truncation_dim, rel_tol, abs_tol, is_TI=False):
    ''' This function only computes the HVP without loss and gradient. '''
    
    # Compute Euclidean derivatives
    cost, grad, hvp = compute_loss_gradient_hvp_F(
        ket0, bra0, Gs, Zs, qubits, first_layer_increasing, truncation_dim, rel_tol, abs_tol, is_TI
        )
   
   # Project to Riemannian manifold
    riem_hvp = postprocess_euclidean_hvp(Gs, Zs, grad, hvp)

    return riem_hvp

def compute_riemannian_hvp_HST(
        ket0, bra0, Gs, Zs, qubits, first_layer_increasing, truncation_dim, rel_tol, abs_tol, is_TI=False):
    ''' This function only computes the HVP without loss and gradient. '''
    
    # Compute Euclidean derivatives
    cost, grad, hvp = compute_loss_gradient_hvp_HST(
        ket0, bra0, Gs, Zs, qubits, first_layer_increasing, truncation_dim, rel_tol, abs_tol, is_TI
        )
   
   # Project to Riemannian manifold
    riem_hvp = postprocess_euclidean_hvp(Gs, Zs, grad, hvp)

    return riem_hvp

def compute_loss_HST_loop(psis, phis, Gs, qubits, first_layer_increasing, 
                          truncation_dim, rel_tol, abs_tol, is_TI=False):
    ''' Compute the loss for n_samples > n_cpus '''
    N = psis.shape[0]
    num_devices = jax.local_device_count()
    
    idx_per_layer = map_list_of_gates_to_layers(qubits)
    flat_idxs = jnp.array(flatten(idx_per_layer))
    layer_ends = jnp.array(gate_is_at_boundary(idx_per_layer))

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

    # Define the function for n_cpu samples

    if is_TI: core_func = _compute_loss_HST_core_TI
    else: core_func = _compute_loss_HST_core
    pmapped_fn = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None),
        out_axes=0,
        axis_name="d",
        static_broadcasted_argnums=(6, 7, 8, 9),
        batched=True
    )
    fid = pmapped_fn(
            batched_psis, batched_phis, Gs, qubits, 
            flat_idxs, layer_ends, first_layer_increasing, 
            truncation_dim, rel_tol, abs_tol
        ) 
    # reduce over all axes
    return jnp.mean(fid)

def compute_riemannian_loss_gradient_HST_loop(
        psis, phis, Gs, qubits, first_layer_increasing, 
        truncation_dim, rel_tol, abs_tol, is_TI=False
        ):
    ''' Compute the loss and Riemannian gradient for n_samples > n_cpus '''
    N = psis.shape[0]
    num_devices = jax.local_device_count()
    
    # ------ Non-jittable preprocessing ------
    idx_per_layer_fw = map_list_of_gates_to_layers(qubits)
    flat_idxs_fw = jnp.array(flatten(idx_per_layer_fw))
    layer_ends_fw = jnp.array(gate_is_at_boundary(idx_per_layer_fw))
    
    idx_per_layer_bw = idx_per_layer_fw[::-1]
    flat_idxs_bw = jnp.array(flat_idxs_fw[::-1])
    layer_ends_bw = jnp.array(gate_is_at_boundary(idx_per_layer_bw))
    
    nlayer_is_even = bool(len(idx_per_layer_bw[0]))
    left_canonical_init = (nlayer_is_even == first_layer_increasing)

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

    if is_TI: core_func = _compute_loss_gradient_HST_core_TI
    else: core_func = _compute_loss_gradient_HST_core
    
    pmapped_fn = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(0, 0),
        axis_name="d",
        static_broadcasted_argnums=(8, 9, 10, 11, 12),
        batched=True
    )
    
    cost_HS, grad_HS = pmapped_fn(
        batched_psis, batched_phis, Gs, qubits, 
        flat_idxs_fw, layer_ends_fw, flat_idxs_bw, layer_ends_bw,
        first_layer_increasing, left_canonical_init, truncation_dim, 
        rel_tol, abs_tol
    )
    
    # Average over all samples
    cost_mean = jnp.mean(cost_HS)
    grad_mean = jnp.mean(grad_HS, axis=(0, 1))
    
    # Project gradient to Riemannian manifold
    riem_grad = project_to_tangent_space(Gs, grad_mean)
    
    return cost_mean, riem_grad

def compute_riemannian_loss_gradient_hvp_HST_loop(
        psis, phis, Gs, Zs, qubits, first_layer_increasing, 
        truncation_dim, rel_tol, abs_tol, is_TI=False
        ):
    ''' Compute the loss, Riemannian gradient, and HVP for n_samples > n_cpus '''
    N = psis.shape[0]
    num_devices = jax.local_device_count()
    
    # ------ Non-jittable preprocessing ------
    idx_per_layer_fw = map_list_of_gates_to_layers(qubits)
    flat_idxs_fw = jnp.array(flatten(idx_per_layer_fw))
    layer_ends_fw = jnp.array(gate_is_at_boundary(idx_per_layer_fw))
    
    idx_per_layer_bw = idx_per_layer_fw[::-1]
    flat_idxs_bw = jnp.array(flat_idxs_fw[::-1])
    layer_ends_bw = jnp.array(gate_is_at_boundary(idx_per_layer_bw))
    
    even_nlayers = bool(len(idx_per_layer_bw[0])!=0)
    left_canonical_init = (even_nlayers == first_layer_increasing)

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

    if is_TI: core_func = _compute_loss_gradient_hvp_HST_core_TI
    else: core_func = _compute_loss_gradient_hvp_HST_core
    
    pmapped_fn = pmap_maybe_batched(
        core_func,
        in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None, None, None),
        out_axes=(0, 0, 0),
        axis_name="d",
        static_broadcasted_argnums=(9, 10, 11, 12, 13),
        batched=True
    )
    
    cost_HS, grad_HS, hvp_HS = pmapped_fn(
        batched_psis, batched_phis, Gs, Zs, qubits,
        flat_idxs_fw, layer_ends_fw, flat_idxs_bw, layer_ends_bw,
        first_layer_increasing, left_canonical_init, truncation_dim, 
        rel_tol, abs_tol
    )
    
    # Average over all samples
    cost_mean = jnp.mean(cost_HS)
    grad_mean = jnp.mean(grad_HS, axis=(0, 1))
    hvp_mean = jnp.mean(hvp_HS, axis=(0, 1))
    
    # Project to Riemannian manifold
    riem_grad, riem_hvp = postprocess_euclidean_loss_gradient_hvp(Gs, Zs, grad_mean, hvp_mean)
    
    return cost_mean, riem_grad, riem_hvp

def pmap_maybe_batched(f, in_axes, out_axes, axis_name, static_broadcasted_argnums, batched=False):
    if batched:
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