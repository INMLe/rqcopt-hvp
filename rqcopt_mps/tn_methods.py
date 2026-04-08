from functools import partial

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from .mps import (
    canonicalize_to_center, 
    right_canonicalize_local_tensor, 
    left_canonicalize_local_tensor, 
    ensure_canonical_center_corresponds_to_gate,
    split_tensor_into_local_MPS_tensors_left_canonical, 
    split_tensor_into_local_MPS_tensors_right_canonical
    )
from .util import mask_to_zeros


#region ========================== Local tensors ==========================

def merge_gate_with_local_mps_tensors_left_canonicalized(
        A_i, A_j, G, truncation_dim, rel_tol, abs_tol
        ):
    '''
    A_i, A_j local tensors of MPS, tensor representation of gate
    '''
    # Merge two-qubit gate with A_i and A_j with optimized contraction path
    B_ = jnp.einsum('aim,klij->akljm', A_i, G)
    B = jnp.einsum('akljm,mjb->aklb', B_, A_j)

    # Split again
    B1, B2 = split_tensor_into_local_MPS_tensors_left_canonical(
        B, truncation_dim, rel_tol, abs_tol
        )
    return B1, B2

def merge_gate_with_local_mps_tensors_right_canonicalized(
        A_i, A_j, G, truncation_dim, rel_tol, abs_tol
        ):
    '''
    A_i, A_j local tensors of MPS, tensor representation of gate
    '''
    # Merge two-qubit gate with A_i and A_j with optimized contraction path
    B_ = jnp.einsum('aim,klij->akljm', A_i, G)
    B = jnp.einsum('akljm,mjb->aklb', B_, A_j)

    # Split again
    B1, B2 = split_tensor_into_local_MPS_tensors_right_canonical(
        B, truncation_dim, rel_tol, abs_tol
        )

    return B1, B2

#endregion


#region ========================== Merge gate with MPS ==========================

def merge_gate_with_mps_left_canonicalized(
        mps, gate, gate_qubits, truncation_dim, rel_tol, abs_tol
        ):
    '''
    Merge a two-qubit gate into an MPS with truncation. Canonical center has to be at 
    location of gate. After merging and truncation push the canonical
    center to the next local tensor in increasing direction (which was not affected by the gate).

    In order to be jittable, gate_qubits need to be jax.array!

    MPS of shape (nsites, chi, p, chi)
    '''

    i, j = gate_qubits
    A_i, A_j = mps[i], mps[j]

    B1, B2 = merge_gate_with_local_mps_tensors_left_canonicalized(
        A_i, A_j, gate, truncation_dim, rel_tol, abs_tol
        )

    # Left-canonicalize B2
    B2, R = left_canonicalize_local_tensor(B2)

    # Push the canonical center to the next local tensor (increasing direction)
    B_next = jnp.einsum('ij,jkl->ikl', R, mps[j+1])

    # Create the updated MPS
    new_canonical_center = j+1
    #mps_updated = mps.copy()
    mps = mps.at[i].set(B1)
    mps = mps.at[j].set(B2)
    mps = mps.at[new_canonical_center].set(B_next)

    return mps, jnp.int32(new_canonical_center)

def merge_gate_with_mps_right_canonicalized(
        mps, gate, gate_qubits, truncation_dim, rel_tol, abs_tol
        ):
    '''
    Merge a two-qubit gate into an MPS with truncation. Canonical center has to be at 
    location of gate. After merging and truncation push the canonical
    center to the next local tensor (which was not affected by gate).

    mps: (nsites,chi,p,chi)
    '''

    i,j = gate_qubits
    A_i, A_j = mps[i], mps[j]

    B1, B2 = merge_gate_with_local_mps_tensors_right_canonicalized(
        A_i, A_j, gate, truncation_dim, rel_tol, abs_tol
        )

    # Right canonicalize B1
    B1, R = right_canonicalize_local_tensor(B1)

    # Push the canonical center to the next local tensor (decreasing direction)
    B_prev = jnp.einsum('ijk,kl->ijl', mps[i-1], R)

    # Create the updated MPS
    new_canonical_center = i-1
    #mps_updated = mps.copy()
    mps = mps.at[new_canonical_center].set(B_prev)
    mps = mps.at[i].set(B1)
    mps = mps.at[j].set(B2)

    return mps, jnp.int32(new_canonical_center)

@partial(jax.jit, static_argnames=("truncation_dim","rel_tol","abs_tol"))
def merge_gate_with_mps_wrapper(
    As, gate, gate_qubits, left_canonical, truncation_dim, rel_tol, abs_tol
    ):
    L = As.shape[0]
    q0 = gate_qubits[0]

    # Choose which kernel to use
    is_left_boundary  = (q0 == 0)
    is_right_boundary = (q0 == (L - 2))

    # left_canonical  AND NOT right_boundary  → left
    # left_canonical  AND     right_boundary  → right
    # NOT left_canonical AND  left_boundary   → left
    # NOT left_canonical AND NOT left_boundary → right
    use_left = jnp.where(left_canonical, ~is_right_boundary, is_left_boundary)

    def do_left(args):
        As_, gate_, qubits_ = args
        return merge_gate_with_mps_left_canonicalized(As_, gate_, qubits_, truncation_dim, rel_tol, abs_tol)

    def do_right(args):
        As_, gate_, qubits_ = args
        return merge_gate_with_mps_right_canonicalized(As_, gate_, qubits_, truncation_dim, rel_tol, abs_tol)

    As_updated, canonical_center = jax.lax.cond(
        use_left, do_left, do_right, (As, gate, gate_qubits)
    )
    As_updated = mask_to_zeros(As_updated, threshold=1e-16)
    return As_updated, canonical_center

#endregion


def merge_layer_with_mps(
        mps, layer, qubits_in_layer, increasing_merging_order=None, 
        caching=False, truncation_dim=128, rel_tol=1e-16, abs_tol=1e-16
        ):
    '''
    If we cache all intermediate states, the order of gate merging is important.
    For the bra intermediate states, we need to do a decreasing sweep.
    For the ket intermediate states, we need to do an increasing sweep.

    If we want to store all intermediate states, each layer has to be merged in an
    increasing order/ decreasing order for kets/bras.
    '''
    if caching: 
        intermediate_mps = [mps]
    if increasing_merging_order:
        first_gate_qubits = qubits_in_layer[0]
        layer_iter = layer
        qubits_iter = qubits_in_layer
        left_canonical = True
        canonical_center = 0
    else:
        gate = layer[-1]
        first_gate_qubits = qubits_in_layer[-1]
        layer_iter = layer[::-1]
        qubits_iter = qubits_in_layer[::-1]
        left_canonical = False
        canonical_center = len(mps)-1
       
    mps = ensure_canonical_center_corresponds_to_gate(
        mps, first_gate_qubits, canonical_center
        )
    for gate, gate_qubits in zip(layer_iter,qubits_iter):
        mps, _ = merge_gate_with_mps_wrapper(
            mps, gate, gate_qubits, left_canonical, truncation_dim, rel_tol, abs_tol
            )
        if caching: intermediate_mps += [mps]
    if caching: return intermediate_mps
    else: return mps

def compute_all_intermediate_states(
        gates_per_layer, qubits_per_layer, phi_0, 
        increasing_merging_order, truncation_dim, 
        rel_tol, abs_tol
        ):
    '''
    Compute $\psi_l$ (original states) by setting increasing_merging_order=True and gates_per_layer are original gates.
    Compute $\phi_l$ (TEBD states) by setting increasing_merging_order=False and gates_per_layer are transposed gates.
    '''
    
    # Initilialize edge environments
    phi, intermediate_states = phi_0, [phi_0]

    # Merge all gates of the brickwall circuit into the initial state
    if increasing_merging_order:  
        layer_iterator=gates_per_layer
        qubits_iterator=qubits_per_layer
    else: 
        layer_iterator=reversed(gates_per_layer)
        qubits_iterator=reversed(qubits_per_layer)
    for layer,qubits_in_layer in zip(layer_iterator,qubits_iterator):
        # At the beginning of a layer merge we need to set the canonical center correctly
        first_affected_qubit = qubits_in_layer[0][0]
        if increasing_merging_order:
            center=1 if first_affected_qubit!=0 else 0
        else: 
            center=phi.nsites-2 if first_affected_qubit!=0 else len(phi)-1            
        phi = canonicalize_to_center(phi, center)
        inter_state = merge_layer_with_mps(
            phi, layer, qubits_in_layer, increasing_merging_order, 
            caching=True, truncation_dim=truncation_dim, 
            rel_tol=rel_tol, abs_tol=abs_tol
            )
        intermediate_states += inter_state[1:]
        phi = inter_state[-1]
    return intermediate_states

def compute_all_intermediate_states_TI(
    Gs, qubits_per_layer, phi_0, increasing_merging_order, 
    truncation_dim, rel_tol, abs_tol
):
    """
    TI Version:
    gates_per_layer: [Gate_Layer1, Gate_Layer2, ..., Gate_LayerL]
    qubits_per_layer: [[(q1,q2), (q3,q4)], [(q0,q1), (q2,q3)], ...]
    """
    
    phi, intermediate_states = phi_0, [phi_0]

    # Set up iterators
    # We zip them to keep gates matched with their qubit layout
    combined = list(zip(Gs, qubits_per_layer))
    
    if not increasing_merging_order: 
        combined = reversed(combined)

    # Layer sweep
    for layer_gate, qubits_in_layer in combined:
        
        # Determine the canonical center for the start of the layer sweep
        first_affected_qubit = qubits_in_layer[0][0]
        
        if increasing_merging_order:
            center = 1 if first_affected_qubit != 0 else 0
        else: 
            # Backwards pass canonicalization logic
            center = phi.nsites - 2 if first_affected_qubit != 0 else len(phi) - 1            
        
        phi = canonicalize_to_center(phi, center)

        # Apply the layer
        # Since merge_layer_with_mps likely expects a list of gates (one per pair),
        # we broadcast the single 'layer_gate' to match the number of qubit pairs.
        gates_to_apply = [layer_gate] * len(qubits_in_layer)

        inter_state = merge_layer_with_mps(
            phi, 
            gates_to_apply, 
            qubits_in_layer, 
            increasing_merging_order, 
            caching=True, 
            truncation_dim=truncation_dim,
            rel_tol=rel_tol, 
            abs_tol=abs_tol
        )
        
        # Append all states generated within this layer sweep (except the first redundant one)
        intermediate_states += inter_state[1:]
        phi = inter_state[-1]

    return intermediate_states