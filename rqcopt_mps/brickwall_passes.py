import jax
import jax.numpy as jnp
from jax import config

from .mps import (
    canonicalize_to_center,
    compute_partial_inner_product_from_mps,
    compute_inner_product_from_mps,
)
from .tn_augmented import (
    apply_aug_gate,
    partial_inner_product_tangent_ket,
    partial_inner_product_tangent_bra,
)
from .tn_methods import merge_gate_with_mps_wrapper

config.update("jax_enable_x64", True)


# ----------------------------------- FORWARD PASS -----------------------------------

def forward_pass_first_order(
    psi0, Gs, qubits, flat_idxs, layer_ends,
    first_layer_increasing, truncation_dim, rel_tol, abs_tol,
    is_TI=False
):
    """
    Forward pass of psi0 through the circuit; returns only the final state.

    is_TI: Gs holds one gate per layer; otherwise one per gate.
    """
    left_canonical = jnp.bool_(first_layer_increasing)
    psi_init = canonicalize_to_center(psi0, qubits[flat_idxs[0]][0])

    def body(carry, t):
        psi, left_can, layer_idx = carry
        idx = flat_idxs[t]
        g = layer_idx if is_TI else idx

        psi, _ = merge_gate_with_mps_wrapper(
            psi, Gs[g], qubits[idx], left_can,
            truncation_dim, rel_tol, abs_tol
        )

        # Switch direction and move to the next layer if the layer ends
        is_layer_end = layer_ends[t]
        left_can = jnp.logical_xor(left_can, is_layer_end)
        layer_idx = layer_idx + jnp.int32(is_layer_end)
        return (psi, left_can, layer_idx), None

    (psi_final, _, _), _ = jax.lax.scan(
        body, (psi_init, left_canonical, jnp.int32(0)),
        jnp.arange(flat_idxs.shape[0], dtype=jnp.int32)
    )
    return psi_final


def forward_pass_first_order_cached(
    psi0, Gs, qubits, flat_idxs_all, layer_ends_all,
    first_layer_increasing, truncation_dim, rel_tol, abs_tol,
    is_TI=False
):
    """
    Forward pass of psi0 through the circuit, keeping every intermediate state.

    is_TI: Gs holds one gate per layer; otherwise one per gate.

    Returns
        psis: (K+1, nsites, chi, p, chi) tensors of psi after each gate (psis[0] = psi0)
    """
    psi_start = canonicalize_to_center(psi0, qubits[flat_idxs_all[0]][0])
    left_canonical_init = jnp.bool_(first_layer_increasing)

    def body(carry, t):
        psi, left_can, layer_idx = carry
        idx = flat_idxs_all[t]
        g = layer_idx if is_TI else idx

        psi_next, _ = merge_gate_with_mps_wrapper(
            psi, Gs[g], qubits[idx], left_can,
            truncation_dim, rel_tol, abs_tol
        )

        # Switch direction and move to the next layer if the layer ends
        is_layer_end = layer_ends_all[t]
        left_can = jnp.logical_xor(left_can, is_layer_end)
        layer_idx = layer_idx + jnp.int32(is_layer_end)

        return (psi_next, left_can, layer_idx), psi_next

    _, psis_stacked = jax.lax.scan(
        body, (psi_start, left_canonical_init, jnp.int32(0)),
        jnp.arange(flat_idxs_all.shape[0], dtype=jnp.int32)
    )

    # History: [original state] + [states after each gate]
    return jnp.concatenate([psi0[None, ...], psis_stacked], axis=0)


# ----------------------------------- BACKWARD PASS -----------------------------------

def backward_pass_first_order(
    phi0, psis, GTs, qubits, flat_idxs, layer_ends,
    left_canonical_init, truncation_dim, rel_tol, abs_tol,
    is_TI=False
):
    """
    Backward pass of phi0 through the transposed circuit.
    flat_idxs / layer_ends are in backward order. psis come from
    forward_pass_first_order_cached.

    is_TI: GTs holds one gate per layer, and the contributions of all gates
    in a layer are summed.

    Returns overlap and grad_F (Frobenius convention).
    """
    left_canonical = jnp.bool_(left_canonical_init)
    last_layer = jnp.int32(GTs.shape[0] - 1)

    def body(carry, t):
        phi, left_can, layer_idx = carry
        idx = flat_idxs[t]
        q = qubits[idx]
        g = layer_idx if is_TI else idx

        # Partial derivative with the forward state before this gate
        grad_val = compute_partial_inner_product_from_mps(
            phi, psis[-t - 2], q
        )

        # Apply the transposed gate to phi
        phi_next, _ = merge_gate_with_mps_wrapper(
            phi, GTs[g], q, left_can,
            truncation_dim, rel_tol, abs_tol
        )

        # Switch direction and move to the previous layer if the layer ends
        is_layer_end = layer_ends[t]
        left_can = jnp.logical_xor(left_can, is_layer_end)
        layer_idx = layer_idx - jnp.int32(is_layer_end)
        return (phi_next, left_can, layer_idx), (g, grad_val)

    (phi_final, _, _), (slots, grad_vals) = jax.lax.scan(
        body, (phi0, left_canonical, last_layer),
        jnp.arange(flat_idxs.shape[0], dtype=jnp.int32)
    )

    # One slot per gate (non-TI) or per layer (TI, contributions are summed)
    grad_F = jnp.zeros_like(GTs).at[slots].add(grad_vals)
    overlap = compute_inner_product_from_mps(phi_final, psis[0])
    return overlap, -grad_F.conj()


# -------------------------- SECOND ORDER --------------------------

def forward_pass_second_order(
    psi0, Gs, Zs, qubits, flat_idxs, layer_ends,
    first_layer_increasing, truncation_dim, rel_tol, abs_tol,
    is_TI=False, fill_slack=True
):
    """
    Forward pass for psi and its tangent along Zs.

    is_TI: Gs and Zs hold one gate per layer; otherwise one per gate.

    Returns
        psis: (K+1, nsites, chi, p, chi) tensors of psi after each gate (psis[0] = psi0)
        Bs:   (K+1, nsites, chi, p, chi) tangent tensors; (psis[k], Bs[k]) represents
              (psi^[k], dpsi^[k]) in the same gauge
        errs: (K,) dropped norm at each gate
    """
    left_canonical = jnp.bool_(first_layer_increasing)
    psi_start = canonicalize_to_center(psi0, qubits[flat_idxs[0]][0])
    B_start = jnp.zeros_like(psi_start)

    def body(carry, inputs):
        As, Bs, left_can, layer_idx = carry
        idx, is_layer_end = inputs
        g = layer_idx if is_TI else idx
        As, Bs, _, err = apply_aug_gate(
            As, Bs, Gs[g], Zs[g], qubits[idx], left_can,
            truncation_dim, rel_tol, abs_tol, fill_slack
            )
        left_can = jnp.logical_xor(left_can, is_layer_end)
        layer_idx = layer_idx + jnp.int32(is_layer_end)
        return (As, Bs, left_can, layer_idx), (As, Bs, err)

    _, (As_stacked, Bs_stacked, errs) = jax.lax.scan(
        body, (psi_start, B_start, left_canonical, jnp.int32(0)), (flat_idxs, layer_ends)
        )

    psis = jnp.concatenate([psi0[None], As_stacked], axis=0)
    Bs = jnp.concatenate([jnp.zeros_like(psi0)[None], Bs_stacked], axis=0)
    return psis, Bs, errs


def backward_pass_second_order(
    phi0, psis, psi_Bs, GTs, ZTs, qubits, flat_idxs, layer_ends,
    left_canonical_init, truncation_dim, rel_tol, abs_tol,
    is_TI=False, fill_slack=True
):
    """
    Backward pass with augmented backward states (phi, dphi).
    flat_idxs / layer_ends are in backward order. psis, psi_Bs come from forward_pass_second_order.

    is_TI: GTs and ZTs hold one gate per layer, and the contributions of all gates
    in a layer are summed.

    Returns overlap, grad_F, hvp_F (Frobenius convention) and
    the dropped norms of the backward states.
    """
    K = flat_idxs.shape[0]
    phi_start = canonicalize_to_center(phi0, qubits[flat_idxs[0]][0])
    dphi_start = jnp.zeros_like(phi_start)

    def body(carry, inputs):
        phi, dphi, left_can, layer_idx = carry
        idx, is_layer_end, t = inputs
        q = qubits[idx]
        g = layer_idx if is_TI else idx

        # Forward state before this gate
        psi_A, psi_B = psis[K - 1 - t], psi_Bs[K - 1 - t]

        grad_val = compute_partial_inner_product_from_mps(phi, psi_A, q)
        hvp_val = (partial_inner_product_tangent_ket(phi, psi_A, psi_B, q)
                   + partial_inner_product_tangent_bra(phi, dphi, psi_A, q))

        phi, dphi, _, err = apply_aug_gate(
            phi, dphi, GTs[g], ZTs[g], q, left_can,
            truncation_dim, rel_tol, abs_tol, fill_slack
            )
        left_can = jnp.logical_xor(left_can, is_layer_end)
        layer_idx = layer_idx - jnp.int32(is_layer_end)
        return (phi, dphi, left_can, layer_idx), (g, grad_val, hvp_val, err)

    ts = jnp.arange(K, dtype=jnp.int32)
    last_layer = jnp.int32(GTs.shape[0] - 1)
    (phi, _, _, _), (slots, grad_vals, hvp_vals, errs) = jax.lax.scan(
        body, (phi_start, dphi_start, jnp.bool_(left_canonical_init), last_layer),
        (flat_idxs, layer_ends, ts)
        )

    # One slot per gate (non-TI) or per layer (TI, contributions are summed)
    grad_F = jnp.zeros_like(GTs).at[slots].add(grad_vals)
    hvp_F = jnp.zeros_like(GTs).at[slots].add(hvp_vals)
    overlap = compute_inner_product_from_mps(phi, psis[0])
    return overlap, -grad_F.conj(), -hvp_F.conj(), errs
