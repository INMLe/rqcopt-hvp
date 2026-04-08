import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .mps import (
    batched_decreasing_RQ_sweep, 
    batched_increasing_SVD_sweep_normalize
)
from .brickwall_circuit import (
    get_unitary_of_layer, 
    get_initial_gates
)
from .brickwall_passes import (
    forward_pass_first_order, 
    forward_pass_first_order_TI
)


def create_tebd_circuit(t, n_sites, n_repetitions, degree, hamiltonian='ising-1d', is_TI=False, **kwargs):
    ''' Createa BrickwallCircuit instance for TEBD. '''
    kwargs['t'] = t
    kwargs['n_repetitions'] = n_repetitions
    kwargs['degree'] = degree
    kwargs['hamiltonian'] = hamiltonian
    layers, qubits = get_initial_gates(n_sites, n_id_layers=0, use_tebd=True, is_TI=is_TI, **kwargs)
    return layers, qubits

def run_tebd(psi, Gs, qubits, flat_idxs, layer_ends, first_layer_increasing, 
             truncation_dim, rel_tol, abs_tol, is_TI=False):
    
    num_total = psi.shape[0]  # Total batch size
    n_cpus = jax.device_count() # N available CPUs
    
    # Choose the appropriate function
    pass_func = forward_pass_first_order_TI if is_TI else forward_pass_first_order
    
    # Define the pmapped function
    parallel_pass = jax.pmap(
        pass_func,
        in_axes=(0, None, None, None, None, None, None, None, None),
        out_axes=0,
        axis_name="d",
        static_broadcasted_argnums=(5, 6, 7, 8),
    )

    results = []
    
    # Loop over the batch in chunks of size n_cpus
    for i in range(0, num_total, n_cpus):
        # Slice the current chunk of states
        psi_chunk = psi[i : i + n_cpus]
        
        # If the last chunk is smaller than n_cpus, pmap might complain 
        # unless you handle padding or use jax.vmap inside.
        # But for exact CPU mapping, we ensure we only pass what we have devices for:
        actual_chunk_size = psi_chunk.shape[0]
        
        if actual_chunk_size == n_cpus:
            out_chunk = parallel_pass(
                psi_chunk, Gs, qubits, flat_idxs, layer_ends, 
                first_layer_increasing, truncation_dim, rel_tol, abs_tol
            )
        else:
            # Handle the remainder (last chunk) using vmap or a smaller pmap
            # Alternatively, pad the chunk to n_cpus and slice it back later
            out_chunk = jax.vmap(pass_func, in_axes=(0,)+(None,)*8)(
                psi_chunk, Gs, qubits, flat_idxs, layer_ends, 
                first_layer_increasing, truncation_dim, rel_tol, abs_tol
            )
            
        results.append(out_chunk)

    # Concatenate all chunks back into one batched MPS
    psi_out = jnp.concatenate(results, axis=0)

    # Normalize TEBD states (keep these batched operations as they are)
    psi_out = batched_decreasing_RQ_sweep(psi_out, psi_out.shape[1], 0)
    psi_out = batched_increasing_SVD_sweep_normalize(psi_out, truncation_dim, rel_tol, abs_tol)

    return psi_out

def run_tebd_matrix(psi, circuit, circuit_qubits, nqubits):
    phi_tebd = psi.copy()
    for layer, qubits in zip(circuit, circuit_qubits):
        U = get_unitary_of_layer(layer, qubits, nqubits)
        phi_tebd = U@phi_tebd
    return phi_tebd