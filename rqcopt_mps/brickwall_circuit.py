from typing import NamedTuple

import jax.numpy as jnp
from jax import config

from .spin_systems import (
    get_brickwall_trotter_gates_spin_chain,
    get_brickwall_trotter_gates_spin_chain_TI,
)

config.update("jax_enable_x64", True)


def get_matrices(tensors):
    return tensors.reshape((-1, 4, 4))


def get_tensors(matrices):
    return matrices.reshape((-1, 2, 2, 2, 2))


def get_initial_gates(n_sites, is_TI, **kwargs):
    hamiltonian = kwargs['hamiltonian']
    t = kwargs['t']
    n_repetitions = kwargs['n_repetitions']
    degree = kwargs['degree']
    del kwargs['hamiltonian']
    del kwargs['t']
    del kwargs['n_repetitions']
    del kwargs['degree']

    if is_TI:
        Glist_start, qubits = get_brickwall_trotter_gates_spin_chain_TI(
            t, n_sites, n_repetitions, degree, hamiltonian, **kwargs
            )
    else:
        Glist_start, qubits = get_brickwall_trotter_gates_spin_chain(
            t, n_sites, n_repetitions, degree, hamiltonian, **kwargs
            )

    return Glist_start, qubits


def map_list_of_gates_to_layers(qubits):
    '''
    This function returns a list of list of indices per layer.
    '''
    qubits = qubits.tolist()
    layers, current_layer, occupied_qubits = [], [], set()

    for i, q in enumerate(qubits):
        q1, q2 = q
        # Start new layer if qubit already in use
        if q1 in occupied_qubits or q2 in occupied_qubits:
            layers.append(current_layer)
            current_layer = []
            occupied_qubits.clear()
        # Add index
        current_layer.append(i)
        occupied_qubits.update((q1, q2))

    if current_layer:
        layers.append(current_layer)

    # It is useful to have an even-lenth list
    if len(layers)%2: layers.append([])

    return layers


def gate_is_at_boundary(idx_per_layer):
    end = [k == (len(layer) - 1) for layer in idx_per_layer for k in range(len(layer))]
    return jnp.array(end, dtype=jnp.bool_)


class CircuitLayout(NamedTuple):
    """
    Order in which the forward and backward passes apply the gates.
    It only depends on the qubit pairs, so compute it once with get_circuit_layout
    and pass it to all cost and derivative functions.
    """
    qubits: jnp.ndarray          # (n_gates, 2) qubit pair of each gate
    flat_idxs_fw: jnp.ndarray    # gate indices in forward order
    layer_ends_fw: jnp.ndarray   # True for the last gate of each layer (forward)
    flat_idxs_bw: jnp.ndarray    # gate indices in backward order
    layer_ends_bw: jnp.ndarray   # True for the last gate of each layer (backward)
    first_layer_increasing: bool  # Forward pass starts with an increasing sweep
    left_canonical_init: bool    # Backward pass starts with an increasing sweep


def first_layer_is_increasing(qubits, idx_per_layer):
    '''
    True if the gates of the first layer are listed from left to right, so the
    forward pass has to start with an increasing sweep.
    A first layer with a single gate counts as increasing.
    '''
    first_layer = idx_per_layer[0]
    if len(first_layer) < 2: return True
    return bool(qubits[first_layer[1]][0] > qubits[first_layer[0]][0])


def get_circuit_layout(qubits):
    qubits = jnp.asarray(qubits)
    idx_per_layer_fw = map_list_of_gates_to_layers(qubits)
    idx_per_layer_bw = idx_per_layer_fw[::-1]
    flat_idxs_fw = jnp.asarray([i for layer in idx_per_layer_fw for i in layer])

    # The sweep direction follows from the gate order of the circuit
    first_layer_increasing = first_layer_is_increasing(qubits, idx_per_layer_fw)

    # An odd number of layers is padded with an empty layer (see map_list_of_gates_to_layers)
    even_nlayers = len(idx_per_layer_bw[0]) != 0

    return CircuitLayout(
        qubits=qubits,
        flat_idxs_fw=flat_idxs_fw,
        layer_ends_fw=gate_is_at_boundary(idx_per_layer_fw),
        flat_idxs_bw=flat_idxs_fw[::-1],
        layer_ends_bw=gate_is_at_boundary(idx_per_layer_bw),
        first_layer_increasing=first_layer_increasing,
        left_canonical_init=(even_nlayers == first_layer_increasing),
    )
