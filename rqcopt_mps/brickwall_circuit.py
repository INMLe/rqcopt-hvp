import jax.numpy as jnp
from scipy.stats import unitary_group
from jax import config
config.update("jax_enable_x64", True)

from .util import tensor_product
from .spin_systems import (
    get_brickwall_trotter_gates_spin_chain, 
    get_brickwall_trotter_gates_spin_chain_TI
    )


def get_matrices(tensors):
    return tensors.reshape((-1,4,4))

def get_tensors(matrices):
    return matrices.reshape((-1,2,2,2,2))

def get_random_TwoQubitGate():
    return unitary_group.rvs(4).reshape((2,2,2,2))

def get_identity_TwoQubitGate():
    return jnp.eye(4, dtype=jnp.complex128).reshape((2,2,2,2))

def layer_is_odd(qubits_in_layer):
    if qubits_in_layer[0][0] == 0 or qubits_in_layer[-1][0] == 0: odd=True
    elif qubits_in_layer[0][0] == 1 or qubits_in_layer[-1][0] == 1: odd=False
    return odd

def get_random_layers(n_sites, n_layers, first_layer_odd, is_TI=False):
    """
    Generate n_layers layers with two-qubit gates expressed
    as tensors --> alternate between odd and even layers.
    """
    layers, qubits_in_layer = [], []
    odd = first_layer_odd
    N_odd_gates, N_even_gates = n_sites//2, n_sites//2  # Number of gates per layer
    if n_sites%2==0: N_even_gates -= 1
        
    for _ in range(n_layers):
        if odd: n_gates, iqubit = N_odd_gates, 0
        else: n_gates, iqubit = N_even_gates, 1
        if is_TI: layers += [[get_random_TwoQubitGate()]]
        else: layers += [[get_random_TwoQubitGate() for _ in range(n_gates)]]
        _qubits_in_layer = [[iqubit+i,iqubit+i+1] for i in range(0,2*n_gates,2)]
        if not odd: _qubits_in_layer = _qubits_in_layer[::-1]
        qubits_in_layer += [_qubits_in_layer]
        odd = not odd

    return layers, qubits_in_layer


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
        print(len(Glist_start))
    else:
        Glist_start, qubits = get_brickwall_trotter_gates_spin_chain(
            t, n_sites, n_repetitions, degree, hamiltonian, **kwargs
            )
            
    return Glist_start, qubits


def get_unitary_of_layer(layer, qubits_in_layer, nqubits):
    ''' Only works for small systems. Assume brickwall circuit structure. '''
    assert nqubits<=8

    I = jnp.eye(2)
    layer = jnp.asarray(layer).reshape((-1,4,4))
    ops = [gate for gate in layer]
    if qubits_in_layer[0][0]==1: ops = [I] + ops
    if qubits_in_layer[-1][-1]==nqubits-2: ops += [I]

    unitary = tensor_product(ops)
    return unitary


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


def snake_layers(layers):
    for i in range(len(layers)):
        if i % 2 == 1:     # odd layers → reverse
            layers[i] = list(reversed(layers[i]))
    return layers


def map_list_of_gates_to_layers_snaking(qubits):
    layers = map_list_of_gates_to_layers(qubits)
    layers = snake_layers(layers)
    return layers