import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from .tn_methods import merge_gate_with_mps_wrapper
from .mps import (
    canonicalize_to_center,
    canonicalize_from_known_center,
    compute_partial_inner_product_from_mps, 
    compute_inner_product_from_mps
    )
from .mps_addition import add_mps


#  ######################## FUNCTIONS FOR A SINGLE SAMPLE ########################

## ----------------------------------- FORWARD PASS -----------------------------------

def forward_pass_first_order(
    psi0, Gs, qubits, flat_idxs, layer_ends, 
    first_layer_increasing, truncation_dim,
    rel_tol, abs_tol
):
    # ------ Initialization ------
    left_canonical = jnp.bool_(first_layer_increasing)    
    psi_init = canonicalize_to_center(psi0, qubits[flat_idxs[0]][0])

    # ------ Define scan ------
    carry0 = (psi_init, left_canonical)
    def body(carry, t):
        psi, left_canonical_state = carry

        idx = flat_idxs[t]
        G = Gs[idx]
        gate_qubits = qubits[idx]

        # Apply the gate
        psi, _ = merge_gate_with_mps_wrapper(
            psi, G, gate_qubits, left_canonical_state, 
            truncation_dim, rel_tol, abs_tol
        )

        # Switch direction if layer ends
        new_left_canonical = jnp.logical_xor(left_canonical_state, layer_ends[t])

        return (psi, new_left_canonical), None

    # ------ Run the sweep ------
    (psi_final, left_canonical_final), _ = jax.lax.scan(
        body, carry0, jnp.arange(flat_idxs.shape[0], dtype=jnp.int32)
    )
    
    return psi_final


def forward_pass_first_order_TI(
    psi0, Gs, qubits, flat_idxs, layer_ends, 
    first_layer_increasing, truncation_dim, rel_tol, abs_tol
):
    """
    Performs a first-order forward pass, accumulating a single total 
    truncation loss value for the entire sweep.
    """
    left_canonical = jnp.bool_(first_layer_increasing)
    
    # ------ Initialization ------
    psi_init = canonicalize_to_center(psi0, qubits[flat_idxs[0]][0])    
    initial_layer_idx = jnp.int32(0)
    carry0 = (psi_init, left_canonical, initial_layer_idx)

    # ------ Define scan ------
    def body(carry, t):
        psi, left_canonical_state, layer_idx = carry

        # Select gate and qubits for current time step
        G = Gs[layer_idx]
        idx = flat_idxs[t]
        gate_qubits = qubits[idx]

        # Apply the gate and capture the local truncation loss
        psi, _ = merge_gate_with_mps_wrapper(
            psi, G, gate_qubits, left_canonical_state, 
            truncation_dim, rel_tol, abs_tol
        )

        # Update layer logic
        is_end_of_layer = layer_ends[t]
        new_left_canonical = jnp.logical_xor(left_canonical_state, is_end_of_layer)
        new_layer_idx = layer_idx + jnp.int32(is_end_of_layer)

        return (psi, new_left_canonical, new_layer_idx), None

    # ------ Run the sweep ------
    (psi_final, _, _), _ = jax.lax.scan(
        body, carry0, jnp.arange(flat_idxs.shape[0], dtype=jnp.int32)
    )
        
    return psi_final


def forward_pass_first_order_cached(
    psi0, Gs, qubits, flat_idxs_all, layer_ends_all,
    first_layer_increasing, truncation_dim, rel_tol, abs_tol
):
    # ------ Initialization ------
    psi_start = canonicalize_to_center(psi0, qubits[flat_idxs_all[0]][0])
    left_canonical_init = jnp.bool_(first_layer_increasing)
    carry0 = (psi_start, left_canonical_init)

    # ------ Define scan ------
    def body(carry, t):
        psi, left_canonical = carry
        idx = flat_idxs_all[t]
        
        # Apply gate and return the current state as the 'output' of this step
        psi_next, _ = merge_gate_with_mps_wrapper(
            psi, Gs[idx], qubits[idx], left_canonical, 
            truncation_dim, rel_tol, abs_tol
        )

        # Update canonical direction for the next step
        new_left_canonical = jnp.logical_xor(left_canonical, layer_ends_all[t])

        # We return psi_next so it gets stacked into psis_stacked
        return (psi_next, new_left_canonical), psi_next

    # ------ Run sweep ------
    _, psis_stacked = jax.lax.scan(
        body, carry0, jnp.arange(flat_idxs_all.shape[0], dtype=jnp.int32)
    )

    # Reconstruct history: [Original state] + [States after each gate]
    psis = jnp.concatenate([psi0[None, ...], psis_stacked], axis=0)
    
    return psis

def forward_pass_first_order_cached_TI(
    psi0, Gs, qubits, flat_idxs_all, layer_ends_all,
    first_layer_increasing, truncation_dim, rel_tol, abs_tol
):
    # ------ Initialization ------
    psi_start = canonicalize_to_center(psi0, qubits[flat_idxs_all[0]][0])
    left_canonical_init = jnp.bool_(first_layer_increasing)
    carry0 = (psi_start, left_canonical_init, jnp.int32(0))

    # ------ Define scan ------
    def body(carry, t):
        psi, left_canonical, layer_idx = carry
        
        # Get global gate index (for qubits) and the TI gate (for G)
        idx = flat_idxs_all[t]
        G = Gs[layer_idx] 
        
        # Apply gate and capture the truncation loss
        psi_next, _ = merge_gate_with_mps_wrapper(
            psi, G, qubits[idx], left_canonical, 
            truncation_dim, rel_tol, abs_tol
        )

        # Logic to update direction and increment layer index
        is_end_of_layer = layer_ends_all[t]
        new_left_canonical = jnp.logical_xor(left_canonical, is_end_of_layer)
        new_layer_idx = layer_idx + jnp.int32(is_end_of_layer)

        # Yield psi_next and loss for history
        return (psi_next, new_left_canonical, new_layer_idx), psi_next

    # ------ Run sweep ------
    _, psis_stacked = jax.lax.scan(
        body, carry0, jnp.arange(flat_idxs_all.shape[0], dtype=jnp.int32)
    )

    # Reconstruct history: [Original state] + [States after each gate]
    psis = jnp.concatenate([psi0[None, ...], psis_stacked], axis=0)
        
    return psis


def forward_pass_second_order(
    psi0, Gs, Zs, qubits, flat_idxs, layer_ends, 
    first_layer_increasing, truncation_dim, rel_tol, abs_tol
):
    # ------ Initialization ------
    left_canonical = jnp.bool_(first_layer_increasing)
    psi_start = canonicalize_to_center(psi0, qubits[flat_idxs[0]][0])
    Dpsi0 = jnp.zeros_like(psi0)

    # Gate g = 0 (Pre-loop processing)
    idx_g0 = flat_idxs[0]
    Dpsi_g0, _ = merge_gate_with_mps_wrapper(
        psi_start, Zs[idx_g0], qubits[idx_g0], left_canonical, 
        truncation_dim, rel_tol, abs_tol
    )
    psi_g0, cc_psi_g0 = merge_gate_with_mps_wrapper(
        psi_start, Gs[idx_g0], qubits[idx_g0], left_canonical, 
        truncation_dim, rel_tol, abs_tol
    )
    
    # Dpsi_g0 came from merge_gate_with_mps_wrapper, so its cc is the same as cc_psi_g0
    cc_Dpsi_g0 = cc_psi_g0

    # Update direction if the first gate was also the end of a layer
    left_canonical_next = jnp.logical_xor(left_canonical, layer_ends[0])

    carry0 = (psi_g0, Dpsi_g0, cc_psi_g0, jnp.int32(cc_Dpsi_g0), left_canonical_next)

    # ------ Define scan ------
    def body(carry, inputs):
        psi, Dpsi, cc_psi, cc_Dpsi, left_can = carry
        idx, is_layer_end = inputs 

        G = Gs[idx] 
        Z = Zs[idx]
        gate_qubits = qubits[idx]

        # --- Update directional derivative ---
        # Ensure Dpsi is centered where the next gate will be applied
        Dpsi = canonicalize_from_known_center(Dpsi, cc_Dpsi, cc_psi)
        
        # Dpsi_next = G(Dpsi) + Z(psi)
        Dpsi_1, _ = merge_gate_with_mps_wrapper(
            Dpsi, G, gate_qubits, left_can, 
            truncation_dim, rel_tol, abs_tol
        )
        Dpsi_2, cc_add = merge_gate_with_mps_wrapper(
            psi, Z, gate_qubits, left_can, 
            truncation_dim, rel_tol, abs_tol
        )
        Dpsi_next, cc_Dpsi_next = add_mps(Dpsi_1, Dpsi_2, cc_add, truncation_dim, rel_tol, abs_tol)

        # --- Update forward intermediate state ---
        psi_next, cc_psi_next = merge_gate_with_mps_wrapper(
            psi, G, gate_qubits, left_can, 
            truncation_dim, rel_tol, abs_tol
        )

        # Switch direction at end of layer
        new_left_can = jnp.logical_xor(left_can, is_layer_end)

        return (psi_next, Dpsi_next, cc_psi_next, cc_Dpsi_next, new_left_can), (psi_next, Dpsi_next)

    # ------ Run sweep ------
    scan_inputs = (flat_idxs[1:], layer_ends[1:])
    _, (psis_stacked, Dpsis_stacked) = jax.lax.scan(body, carry0, scan_inputs)

    # Reconstruct full history: [t=init] + [g=0] + [g=1...n]
    psis = jnp.concatenate([psi0[None, ...], psi_g0[None, ...], psis_stacked], axis=0)
    Dpsis = jnp.concatenate([Dpsi0[None, ...], Dpsi_g0[None, ...], Dpsis_stacked], axis=0)

    return psis, Dpsis


def forward_pass_second_order_TI(
    psi0, Gs, Zs, qubits, flat_idxs, layer_ends, 
    first_layer_increasing, truncation_dim, rel_tol, abs_tol
):
    """
    Computes psi and Dpsi history while accumulating a single total loss 
    value for the entire circuit sweep.
    """
    # ------ Initialization ------
    left_canonical = jnp.bool_(first_layer_increasing)
    psi_start = canonicalize_to_center(psi0, qubits[flat_idxs[0]][0])
    Dpsi0 = jnp.zeros_like(psi0)
    layer_idx = jnp.int32(0)

    # Initial Gate (g = 0)
    idx_g0 = flat_idxs[0]
    G_g0, Z_g0 = Gs[layer_idx], Zs[layer_idx]
    
    # Update state and derivative
    psi_g0, cc_psi_g0 = merge_gate_with_mps_wrapper(
        psi_start, G_g0, qubits[idx_g0], left_canonical, 
        truncation_dim, rel_tol, abs_tol
    )
    
    Dpsi_g0, _ = merge_gate_with_mps_wrapper(
        psi_start, Z_g0, qubits[idx_g0], left_canonical, 
        truncation_dim, rel_tol, abs_tol
    )

    # Dpsi_g0 came from merge_gate_with_mps_wrapper, so its cc is the same as cc_psi_g0
    cc_Dpsi_g0 = cc_psi_g0

    # Update state for the scan: check if g=0 was the end of layer 0
    is_g0_end = layer_ends[0]
    left_canonical_next = jnp.logical_xor(left_canonical, is_g0_end)
    layer_idx_next = layer_idx + jnp.int32(is_g0_end)

    # Carry: (psi, Dpsi, center, canonical_dir, layer_idx, accumulated_scalar_loss)
    carry0 = (psi_g0, Dpsi_g0, cc_psi_g0, cc_Dpsi_g0, left_canonical_next, layer_idx_next)
    
    # ------ Define scan ------
    def body(carry, inputs):
        psi, Dpsi, cc_psi, cc_Dpsi, left_can, l_idx = carry
        idx, is_layer_end = inputs 

        G, Z = Gs[l_idx], Zs[l_idx]
        gate_qubits = qubits[idx]

        # --- Update directional derivative ---
        Dpsi = canonicalize_from_known_center(Dpsi, cc_Dpsi, cc_psi)
        
        # Update Dpsi (Leibniz Rule) # Dpsi_next = G(Dpsi) + Z(psi)
        Dpsi_1, _ = merge_gate_with_mps_wrapper(
            Dpsi, G, gate_qubits, left_can, truncation_dim, rel_tol, abs_tol
        )
        Dpsi_2, cc_add = merge_gate_with_mps_wrapper(
            psi, Z, gate_qubits, left_can, truncation_dim, rel_tol, abs_tol
        )
        Dpsi_next, cc_Dpsi_next  = add_mps(Dpsi_1, Dpsi_2, cc_add, truncation_dim, rel_tol, abs_tol)

        # Update Primal psi
        psi_next, cc_psi_next = merge_gate_with_mps_wrapper(
            psi, G, gate_qubits, left_can, truncation_dim, rel_tol, abs_tol
        )

        new_left_can = jnp.logical_xor(left_can, is_layer_end)
        new_l_idx = l_idx + jnp.int32(is_layer_end)

        return (psi_next, Dpsi_next, cc_psi_next, cc_Dpsi_next, new_left_can, new_l_idx), \
               (psi_next, Dpsi_next)
    
    # ------ Run sweep ------
    scan_inputs = (flat_idxs[1:], layer_ends[1:])
    _, (psis_stacked, Dpsis_stacked) = jax.lax.scan(body, carry0, scan_inputs)


    # Reconstruct history
    psis = jnp.concatenate([psi0[None, ...], psi_g0[None, ...], psis_stacked], axis=0)
    Dpsis = jnp.concatenate([Dpsi0[None, ...], Dpsi_g0[None, ...], Dpsis_stacked], axis=0)
    
    return psis, Dpsis




## ----------------------------------- BACKWARD PASS -----------------------------------

def backward_pass_first_order(
    phi0, psis, GTs, qubits, flat_idxs, layer_ends, 
    left_canonical_init, truncation_dim, rel_tol, abs_tol
):
    # ------ Initialization ------
    left_canonical = jnp.bool_(left_canonical_init)
    g_start_offset = jnp.int32(1)
    carry0 = (phi0, left_canonical)

    # ------ Define scan ------
    def body(carry, t):
        phi, left_can = carry
        
        # Calculate current position in the psi history (moving backward)
        curr_g = t + g_start_offset 
        idx = flat_idxs[t]
        gate_qubits = qubits[idx]
        
        # Compute the partial derivative (local gradient component)
        # We access the forward history from the end
        grad_val = compute_partial_inner_product_from_mps(
            phi, psis[-curr_g - 1], gate_qubits
        )
        
        # Update the backward state (phi) by applying the transposed gate
        phi_next, _ = merge_gate_with_mps_wrapper(
            phi, GTs[idx], gate_qubits, left_can, 
            truncation_dim, rel_tol, abs_tol
        )

        # Update direction at the end of a layer
        new_left_can = jnp.logical_xor(left_can, layer_ends[t])

        return (phi_next, new_left_can), (idx, grad_val)

    # ------ Run sweep ------
    (phi_final, _), (idxs_stacked, grad_vals_stacked) = jax.lax.scan(
        body,
        carry0,
        jnp.arange(flat_idxs.shape[0], dtype=jnp.int32)
    )

    # Reconstruct the full gradient array based on the indices visited
    grad_F = jnp.zeros_like(GTs).at[idxs_stacked].set(grad_vals_stacked)
    grad_F_out = -grad_F.conj()
    
    # Calculate the final overlap between the unwound state and the initial state
    overlap = compute_inner_product_from_mps(phi_final, psis[0])

    return overlap, grad_F_out

def backward_pass_cached_TI(
    phi0, GTs, qubits, flat_idxs, layer_ends, 
    left_canonical_init, truncation_dim
):
    """
    Computes history of intermediate states and truncation losses during the backward sweep.
    """
    
    # ------ Initialization ------
    # Start at the LAST layer index of the Adjoint Gates
    initial_layer_idx = jnp.int32(GTs.shape[0] - 1)
    carry0 = (phi0, jnp.bool_(left_canonical_init), initial_layer_idx)

    # ------ Define scan ------
    def body(carry, t):
        phi, left_can, l_idx = carry
        
        idx = flat_idxs[t]
        gate_qubits = qubits[idx]
        
        # Apply the Adjoint Gate (GTs) and capture the local truncation loss
        phi_next, _ = merge_gate_with_mps_wrapper(
            phi, GTs[l_idx], gate_qubits, left_can, truncation_dim
        )

        # Logic for layer boundaries
        is_end_of_layer = layer_ends[t]
        new_left_can = jnp.logical_xor(left_can, is_end_of_layer)
        
        # Decrement the layer index when crossing a layer boundary
        new_l_idx = l_idx - jnp.int32(is_end_of_layer)

        # We return phi_next and the loss as the stacked outputs
        return (phi_next, new_left_can, new_l_idx), phi_next

    # ------ Run backward sweep ------
    _, phis_stacked = jax.lax.scan(
        body, 
        carry0, 
        jnp.arange(flat_idxs.shape[0], dtype=jnp.int32)
    )

    # Reconstruct histories: [phi0, gate1_out, gate2_out, ...]
    phis = jnp.concatenate([phi0[None, ...], phis_stacked], axis=0)
    
    return phis


def backward_pass_first_order_TI(
    phi0, psis, GTs, qubits, flat_idxs, layer_ends, 
    left_canonical_init, truncation_dim, rel_tol, abs_tol
):
    # ------ Initialization ------
    left_canonical = jnp.bool_(left_canonical_init)
    g_start_offset = jnp.int32(1)
    
    # We track layer_idx backwards. We start at the LAST layer index.
    initial_layer_idx = jnp.int32(GTs.shape[0] - 1)
    
    carry0 = (phi0, left_canonical, initial_layer_idx)

    # ------ Define scan ------
    def body(carry, t):
        phi, left_can, l_idx = carry
        
        # Calculate current position in the psi history (moving backward)
        curr_g = t + g_start_offset 
        idx = flat_idxs[t]
        gate_qubits = qubits[idx]
        
        # Compute the partial derivative at this specific position
        grad_val = compute_partial_inner_product_from_mps(
            phi, psis[-curr_g - 1], gate_qubits
        )
        
        # Update the backward state (phi) using the TI gate
        phi_next, _ = merge_gate_with_mps_wrapper(
            phi, GTs[l_idx], gate_qubits, left_can, 
            truncation_dim, rel_tol, abs_tol
        )

        # Logic for layer boundaries
        is_end_of_layer = layer_ends[t]
        new_left_can = jnp.logical_xor(left_can, is_end_of_layer)
        
        # Since we are going backwards, we DECREMENT the layer index 
        # when we cross a layer boundary
        new_l_idx = l_idx - jnp.int32(is_end_of_layer)

        # We return l_idx and grad_val to sum them up later
        return (phi_next, new_left_can, new_l_idx), (l_idx, grad_val)

    # ------ Run sweep ------
    (phi_final, _, _), (layer_idxs_stacked, grad_vals_stacked) = jax.lax.scan(
        body,
        carry0,
        jnp.arange(flat_idxs.shape[0], dtype=jnp.int32)
    )

    # Gradient aggregation (summation!)
    grad_F = jnp.zeros_like(GTs).at[layer_idxs_stacked].add(grad_vals_stacked)
    grad_F_out = -grad_F.conj()
    
    # Final overlap calculation
    overlap = compute_inner_product_from_mps(phi_final, psis[0])

    return overlap, grad_F_out


def apply_all_gates_backward_pass_second_order(
        phi_init, Dphi_init, cc_phi_init, cc_Dphi_init, left_canonical_init, g_init,
        psis, Dpsis, GTs, ZTs, qubits, flat_idxs, layer_ends, truncation_dim,
        rel_tol, abs_tol
        ):
    
    # ------ Initialization ------
    carry0 = (phi_init, Dphi_init, cc_phi_init, jnp.int32(cc_Dphi_init), left_canonical_init)

    # ------ Define scan ------
    def body(carry, inputs):
        phi, Dphi, cc_phi, cc_Dphi, left_canonical = carry
        idx, is_layer_end, t = inputs 

        curr_g = t + g_init
        
        gate_qubits = qubits[idx]
        GT = GTs[idx]
        ZT = ZTs[idx]

        # ---- Gradient calculation ----
        grad_val = compute_partial_inner_product_from_mps(phi, psis[-curr_g-1], gate_qubits)
        
        # ---- HVP calculation ----
        hvp_1 = compute_partial_inner_product_from_mps(phi, Dpsis[-curr_g-1], gate_qubits)
        hvp_2 = compute_partial_inner_product_from_mps(Dphi, psis[-curr_g-1], gate_qubits)
        hvp_val = hvp_1 + hvp_2

        # ---- Update Dphi ----
        Dphi = canonicalize_from_known_center(Dphi, cc_Dphi, cc_phi)
        Dphi_1, cc_add = merge_gate_with_mps_wrapper(
            Dphi, GT, gate_qubits, left_canonical, 
            truncation_dim, rel_tol, abs_tol
        )
        Dphi_2, _ = merge_gate_with_mps_wrapper(
            phi, ZT, gate_qubits, left_canonical, 
            truncation_dim, rel_tol, abs_tol
        )
        Dphi, cc_Dphi_new = add_mps(Dphi_1, Dphi_2, cc_add, truncation_dim, rel_tol, abs_tol)

        # ---- Update phi ----
        phi, cc_phi = merge_gate_with_mps_wrapper(
            phi, GT, gate_qubits, left_canonical, 
            truncation_dim, rel_tol, abs_tol
        )

        # Switch direction
        left_canonical = jnp.logical_xor(left_canonical, is_layer_end)

        return (phi, Dphi, cc_phi, cc_Dphi_new, left_canonical), (idx, grad_val, hvp_val)

    # ------ Run sweep ------
    scan_len = flat_idxs.shape[0]
    ts = jnp.arange(scan_len, dtype=jnp.int32)
    scan_inputs = (flat_idxs, layer_ends, ts)
    (phi, Dphi, cc_phi, cc_Dphi, left_canonical), (idxs_stacked, grad_vals, hvp_vals) = jax.lax.scan(
        body,
        carry0,
        scan_inputs,
    )

    return phi, Dphi, cc_phi, cc_Dphi, left_canonical, idxs_stacked, grad_vals, hvp_vals


def backward_pass_second_order(
    phi0, psis, Dpsis, GTs, ZTs, qubits, flat_idxs, 
    layer_ends, left_canonical_init, truncation_dim,
    rel_tol, abs_tol
    ):
    # ------ Initialization ------
    phi = phi0
    Dphi = jnp.zeros_like(phi0)
    
    # Pre-allocate result arrays
    grad_F = jnp.zeros_like(GTs)
    hvp_F  = jnp.zeros_like(GTs)

    # -------- Gate g = n (last gate) --------
    grad_val_g1 = compute_partial_inner_product_from_mps(phi, psis[-2], qubits[-1])
    hvp_val_g1  = compute_partial_inner_product_from_mps(phi, Dpsis[-2], qubits[-1])
    
    grad_F = grad_F.at[-1].set(grad_val_g1)
    hvp_F  = hvp_F.at[-1].set(hvp_val_g1)

    # Update states after g=n
    phi = canonicalize_to_center(phi, qubits[-1][0])
    Dphi, cc_Dphi = merge_gate_with_mps_wrapper(
        phi, ZTs[-1], qubits[-1], left_canonical_init, 
        truncation_dim, rel_tol, abs_tol
    )
    phi, cc_phi = merge_gate_with_mps_wrapper(
        phi, GTs[-1], qubits[-1], left_canonical_init, 
        truncation_dim, rel_tol, abs_tol
    )

    # -------- Loop: g = n-1 ... 2 --------
    phi, Dphi, cc_phi, cc_Dphi, left_canonical, idxs_loop, grads_loop, hvps_loop = \
        apply_all_gates_backward_pass_second_order(
            phi_init=phi, Dphi_init=Dphi, cc_phi_init=cc_phi,
            cc_Dphi_init=cc_Dphi,
            left_canonical_init=jnp.bool_(left_canonical_init), 
            g_init=jnp.int32(2), 
            psis=psis, Dpsis=Dpsis, GTs=GTs, ZTs=ZTs, qubits=qubits,
            flat_idxs=flat_idxs[1:-1],   # Skip g=n, g=1
            layer_ends=layer_ends[1:-1],  # Skip g=n, g=1
            truncation_dim=truncation_dim,
            rel_tol=rel_tol, 
            abs_tol=abs_tol
        )
    
    # Scatter results back into the full arrays
    grad_F = grad_F.at[idxs_loop].set(grads_loop)
    hvp_F  = hvp_F.at[idxs_loop].set(hvps_loop)

    # -------- Gate g = 1 (first gate) --------
    gate_qubits = qubits[0]
    GT = GTs[0]
    ZT = ZTs[0]
    
    grad_val_gn = compute_partial_inner_product_from_mps(phi, psis[0], gate_qubits)
    hvp_val_gn  = compute_partial_inner_product_from_mps(Dphi, psis[0], gate_qubits)

    grad_F = grad_F.at[0].set(grad_val_gn)
    hvp_F  = hvp_F.at[0].set(hvp_val_gn)

    # Update last state
    Dphi = canonicalize_from_known_center(Dphi, cc_Dphi, gate_qubits[0])
    Dphi_1, canonical_center = merge_gate_with_mps_wrapper(
        Dphi, GT, gate_qubits, left_canonical, 
        truncation_dim, rel_tol, abs_tol
    )
    phi = canonicalize_from_known_center(phi, cc_phi, gate_qubits[0])
    Dphi_2, _ = merge_gate_with_mps_wrapper(
        phi, ZT, gate_qubits, left_canonical, 
        truncation_dim, rel_tol, abs_tol
    )
    Dphi, _ = add_mps(Dphi_1, Dphi_2, canonical_center, truncation_dim, rel_tol, abs_tol)
    phi, _ = merge_gate_with_mps_wrapper(
        phi, GT, gate_qubits, left_canonical, 
        truncation_dim, rel_tol, abs_tol
    )

    # -------- Collect results --------
    grad_F_out = -grad_F.conj()
    hvp_F_out  = -hvp_F.conj()
    overlap    = compute_inner_product_from_mps(phi, psis[0])

    return overlap, grad_F_out, hvp_F_out


def apply_all_gates_backward_pass_second_order_TI(
    phi_init, Dphi_init, cc_phi_init, cc_Dphi_init, left_canonical_init, layer_idx_init, g_init,
    psis, Dpsis, GTs, ZTs, qubits, flat_idxs, layer_ends, truncation_dim, rel_tol, abs_tol
):
    # ------ Initialization ------
    carry0 = (phi_init, Dphi_init, cc_phi_init, cc_Dphi_init, jnp.bool_(left_canonical_init), jnp.int32(layer_idx_init))

    # ------ Define scan ------
    def body(carry, inputs):
        phi, Dphi, cc_phi, cc_Dphi, left_canonical, layer_idx = carry
        idx, is_layer_end, t = inputs

        curr_g = t + g_init
        gate_qubits = qubits[idx]

        # TI gates for current layer
        GT = GTs[layer_idx]
        ZT = ZTs[layer_idx]

        # ---- Gradient calculation ----
        grad_val = compute_partial_inner_product_from_mps(
            phi, psis[-curr_g - 1], gate_qubits
        )

        # ---- HVP calculation ----
        hvp_1 = compute_partial_inner_product_from_mps(
            phi, Dpsis[-curr_g - 1], gate_qubits
        )
        hvp_2 = compute_partial_inner_product_from_mps(
            Dphi, psis[-curr_g - 1], gate_qubits
        )
        hvp_val = hvp_1 + hvp_2

        # ---- Update Dphi ----
        Dphi = canonicalize_from_known_center(Dphi, cc_Dphi, cc_phi)
        Dphi_1, cc_add = merge_gate_with_mps_wrapper(
            Dphi, GT, gate_qubits, left_canonical, 
            truncation_dim, rel_tol, abs_tol
        )
        Dphi_2, _ = merge_gate_with_mps_wrapper(
            phi, ZT, gate_qubits, left_canonical, 
            truncation_dim, rel_tol, abs_tol
        )
        Dphi, cc_Dphi_new = add_mps(Dphi_1, Dphi_2, cc_add, truncation_dim, rel_tol, abs_tol)

        # ---- Update phi ----
        phi, cc_phi = merge_gate_with_mps_wrapper(
            phi, GT, gate_qubits, left_canonical, 
            truncation_dim, rel_tol, abs_tol
        )

        # Switch direction
        left_canonical = jnp.logical_xor(left_canonical, is_layer_end)

        # Going backward: move to previous layer when we finish a layer
        layer_idx = layer_idx - jnp.int32(is_layer_end)

        return (phi, Dphi, cc_phi, cc_Dphi_new, left_canonical, layer_idx), (layer_idx + jnp.int32(is_layer_end), grad_val, hvp_val)

    # ------ Run sweep ------
    scan_len = flat_idxs.shape[0]
    ts = jnp.arange(scan_len, dtype=jnp.int32)
    scan_inputs = (flat_idxs, layer_ends, ts)

    (phi, Dphi, cc_phi, cc_Dphi, left_canonical, layer_idx_out), (layer_idxs_stacked, grad_vals, hvp_vals) = jax.lax.scan(
        body, carry0, scan_inputs
    )

    return phi, Dphi, cc_phi, cc_Dphi, left_canonical, layer_idx_out, layer_idxs_stacked, grad_vals, hvp_vals


def backward_pass_second_order_TI(
    phi0, psis, Dpsis, GTs, ZTs, qubits, flat_idxs,
    layer_ends, left_canonical_init, truncation_dim, 
    rel_tol, abs_tol
):
    # ------ Initialization ------
    phi = phi0
    Dphi = jnp.zeros_like(phi0)
    nlayers = GTs.shape[0]

    # Pre-allocate result arrays (PER LAYER)
    grad_F = jnp.zeros_like(GTs)
    hvp_F  = jnp.zeros_like(GTs)

    # We start in the last layer
    layer_idx = jnp.int32(nlayers - 1)
    left_canonical = jnp.bool_(left_canonical_init)

    # -------- Gate g = n (first step in backward schedule) --------
    idx_gn = flat_idxs[0]
    gate_qubits_gn = qubits[idx_gn]

    grad_val_gn = compute_partial_inner_product_from_mps(phi, psis[-2], gate_qubits_gn)
    hvp_val_gn  = compute_partial_inner_product_from_mps(phi, Dpsis[-2], gate_qubits_gn)

    grad_F = grad_F.at[layer_idx].add(grad_val_gn)
    hvp_F  = hvp_F.at[layer_idx].add(hvp_val_gn)

    # Update states after g=n
    phi = canonicalize_to_center(phi, gate_qubits_gn[0])
    Dphi, cc_Dphi = merge_gate_with_mps_wrapper(
        phi, ZTs[layer_idx], gate_qubits_gn, left_canonical, 
        truncation_dim, rel_tol, abs_tol
    )
    phi, cc_phi = merge_gate_with_mps_wrapper(
        phi, GTs[layer_idx], gate_qubits_gn, left_canonical, 
        truncation_dim, rel_tol, abs_tol
    )

    # Update direction/layer after this step (like your forward TI does)
    is_end = layer_ends[0]
    left_canonical = jnp.logical_xor(left_canonical, is_end)
    layer_idx = layer_idx - jnp.int32(is_end)

    # -------- Loop: g = n-1 ... 2 --------
    phi, Dphi, cc_phi, cc_Dphi, left_canonical, layer_idx, layer_idxs_loop, grads_loop, hvps_loop = \
        apply_all_gates_backward_pass_second_order_TI(
            phi_init=phi,
            Dphi_init=Dphi,
            cc_phi_init=cc_phi,
            cc_Dphi_init=cc_Dphi,
            left_canonical_init=left_canonical,
            layer_idx_init=layer_idx,
            g_init=jnp.int32(2),
            psis=psis,
            Dpsis=Dpsis,
            GTs=GTs,
            ZTs=ZTs,
            qubits=qubits,
            flat_idxs=flat_idxs[1:-1],      # Skip g=n and g=1
            layer_ends=layer_ends[1:-1],    # aligned slice
            truncation_dim=truncation_dim,
            rel_tol=rel_tol, 
            abs_tol=abs_tol
        )

    # Aggregate results (TI: SUM per layer)
    grad_F = grad_F.at[layer_idxs_loop].add(grads_loop)
    hvp_F  = hvp_F.at[layer_idxs_loop].add(hvps_loop)

    # -------- Gate g = 1 (last step in backward schedule) --------
    idx_g1 = flat_idxs[-1]
    gate_qubits_g1 = qubits[idx_g1]

    GT = GTs[layer_idx]
    ZT = ZTs[layer_idx]

    grad_val_g1 = compute_partial_inner_product_from_mps(phi, psis[0], gate_qubits_g1)
    hvp_val_g1  = compute_partial_inner_product_from_mps(Dphi, psis[0], gate_qubits_g1)

    grad_F = grad_F.at[layer_idx].add(grad_val_g1)
    hvp_F  = hvp_F.at[layer_idx].add(hvp_val_g1)

    # Update last state (same as original, but TI gate)
    Dphi = canonicalize_from_known_center(Dphi, cc_Dphi, gate_qubits_g1[0])
    Dphi_1, canonical_center = merge_gate_with_mps_wrapper(
        Dphi, GT, gate_qubits_g1, left_canonical, 
        truncation_dim, rel_tol, abs_tol
    )
    phi = canonicalize_from_known_center(phi, cc_phi, gate_qubits_g1[0])
    Dphi_2, _ = merge_gate_with_mps_wrapper(
        phi, ZT, gate_qubits_g1, left_canonical, 
        truncation_dim, rel_tol, abs_tol
    )
    Dphi, _ = add_mps(Dphi_1, Dphi_2, canonical_center, truncation_dim, rel_tol, abs_tol)
    phi, _ = merge_gate_with_mps_wrapper(
        phi, GT, gate_qubits_g1, left_canonical, 
        truncation_dim, rel_tol, abs_tol
    )

    # -------- Collect results --------
    grad_F_out = -grad_F.conj()
    hvp_F_out  = -hvp_F.conj()
    overlap    = compute_inner_product_from_mps(phi, psis[0])

    return overlap, grad_F_out, hvp_F_out


def backward_pass_second_order_intermediate_states_TI(
    phi0, GTs, ZTs, qubits, flat_idxs, layer_ends, 
    left_canonical_init, truncation_dim, rel_tol, abs_tol
):
    """
    Returns the history of backward intermediate states (phi), 
    their directional derivatives (Dphi), and the single total scalar truncation loss.
    """
    # ------ Initialization ------
    phi_start = phi0
    Dphi_start = jnp.zeros_like(phi0)
    
    nlayers = GTs.shape[0]
    initial_layer_idx = jnp.int32(nlayers - 1)
    
    # --- Handle Gate g = 0 (Pre-loop processing) ---
    idx_g0 = flat_idxs[0]
    gate_qubits_g0 = qubits[idx_g0]
    GT_g0 = GTs[initial_layer_idx]
    ZT_g0 = ZTs[initial_layer_idx]

    # phi_1 = GT(phi0)
    phi_g0, cc_phi_g0 = merge_gate_with_mps_wrapper(
        phi_start, GT_g0, gate_qubits_g0, left_canonical_init, 
        truncation_dim, rel_tol, abs_tol
    )
    # Dphi_1 = ZT(phi0) + GT(Dphi0). Since Dphi0 is 0, this is ZT(phi0)
    Dphi_g0, _ = merge_gate_with_mps_wrapper(
        phi_start, ZT_g0, gate_qubits_g0, left_canonical_init, 
        truncation_dim, rel_tol, abs_tol
    )

    is_g0_end = layer_ends[0]
    left_can_next = jnp.logical_xor(left_canonical_init, is_g0_end)
    layer_idx_next = initial_layer_idx - jnp.int32(is_g0_end)

    carry0 = (phi_g0, Dphi_g0, cc_phi_g0, left_can_next, layer_idx_next)

    # ------ Define scan ------
    def body(carry, inputs):
        phi, Dphi, cc_phi, left_can, l_idx = carry
        idx, is_layer_end = inputs

        GT, ZT = GTs[l_idx], ZTs[l_idx]
        gate_qubits = qubits[idx]

        Dphi = canonicalize_to_center(Dphi, cc_phi)
        
        # Update Backward Derivative components (Leibniz Rule)
        Dphi_1, cc_add = merge_gate_with_mps_wrapper(
            Dphi, GT, gate_qubits, left_can, 
            truncation_dim, rel_tol, abs_tol
        )
        Dphi_2, _ = merge_gate_with_mps_wrapper(
            phi, ZT, gate_qubits, left_can, 
            truncation_dim, rel_tol, abs_tol
        )
        # Add components and capture compression loss
        Dphi_next, _ = add_mps(
            Dphi_1, Dphi_2, cc_add, truncation_dim, rel_tol, abs_tol
            )

        # Update Backward State
        phi_next, cc_phi_next = merge_gate_with_mps_wrapper(
            phi, GT, gate_qubits, left_can, truncation_dim, rel_tol, abs_tol
        )

        new_left_can = jnp.logical_xor(left_can, is_layer_end)
        new_l_idx = l_idx - jnp.int32(is_layer_end)

        return (phi_next, Dphi_next, cc_phi_next, new_left_can, new_l_idx), \
               (phi_next, Dphi_next)

    # ------ Run sweep ------
    scan_inputs = (flat_idxs[1:], layer_ends[1:])
    final_carry, (phis_stacked, Dpsis_stacked) = jax.lax.scan(body, carry0, scan_inputs)

    # Reconstruct history     
    phis = jnp.concatenate([phi_start[None, ...], phi_g0[None, ...], phis_stacked], axis=0)
    Dphis = jnp.concatenate([Dphi_start[None, ...], Dphi_g0[None, ...], Dpsis_stacked], axis=0)
    
    return phis, Dphis