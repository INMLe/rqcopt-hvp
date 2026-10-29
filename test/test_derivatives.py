import os

# This needs to be set before importing jax!
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=5"

import jax

# This needs to be set before rqcopt_mps creates arrays at import time.
jax.config.update('jax_platform_name', 'cpu')

import jax.numpy as jnp
import pytest
from jax import random

from rqcopt_mps import (
    brickwall_circuit,
    brickwall_passes,
    objective_function,
    riemannian_manifold,
    tn_helpers,
    tn_methods,
    util,
)
from test_helpers import (
    get_random_layers,
    get_random_mps,
    get_random_mps_stacked,
    get_tangent_vector,
    get_vector,
)

key = random.PRNGKey(0)
keys = random.split(key, 20)

rel_tol, abs_tol = 1e-11, 1e-14

# Every test (except test_get_gates_per_layer) runs for a general brickwall circuit
# (one gate per gate position) and a translation-invariant (TI) one (one gate per layer).
TI_cases = pytest.mark.parametrize("is_TI", [False, True], ids=["general", "TI"])


# ----- Helpers -----

def random_circuit(nsites, nlayers, is_TI, with_direction=False):
    '''
    Random brickwall circuit. Returns the gate tensors Gs, the flat qubit pairs,
    the qubits per layer and, if with_direction, a random direction Zs of the same shape.
    '''
    G_layers, qubits_in_layer = get_random_layers(nsites, nlayers, first_layer_odd=True, is_TI=is_TI)
    Gs = jnp.asarray(util.flatten(G_layers))
    qubits = util.flatten(qubits_in_layer)
    if not with_direction:
        return Gs, qubits, qubits_in_layer
    Z_layers, _ = get_random_layers(nsites, nlayers, first_layer_odd=True, is_TI=is_TI)
    Zs = jnp.asarray(util.flatten(Z_layers))
    return Gs, Zs, qubits, qubits_in_layer


def gate_per_position(gates, qubits_in_layer, is_TI):
    ''' One 4x4 matrix per gate position in circuit order. In the TI case each layer repeats its gate. '''
    gates = gates.reshape((-1, 4, 4))
    if is_TI:
        return [gates[ell] for ell, layer in enumerate(qubits_in_layer) for _ in layer]
    return list(gates)


def apply_circuit_dense(gates, qubits_in_layer, psi, is_TI):
    ''' Apply the circuit gate by gate to the state vector psi. Returns all states, psi first. '''
    nsites = psi.size.bit_length() - 1  # psi has 2**nsites entries
    positions = [q[0] for layer in qubits_in_layer for q in layer]
    states = [psi]
    for G, q in zip(gate_per_position(gates, qubits_in_layer, is_TI), positions):
        op = util.tensor_product([jnp.eye(2**q), G, jnp.eye(2**(nsites-q-2))])
        states.append(op @ states[-1])
    return states


def dense_cost(kind, kets, bras, qubits_in_layer, is_TI):
    '''
    Cost as a function of the gate matrices (n_gates or n_layers, 4, 4), computed with state vectors.
    kind='F': 1 - mean Re<bra|U|ket>, kind='HST': 1 - mean |<bra|U|ket>|^2.
    '''
    def f(gates):
        res = 0
        for ket, bra in zip(kets, bras):
            overlap = jnp.dot(get_vector(bra), apply_circuit_dense(gates, qubits_in_layer, get_vector(ket), is_TI)[-1])
            res += overlap.real if kind == 'F' else jnp.abs(overlap)**2
        return 1 - res/len(kets)
    return f


def dense_cost_gradient_hvp(f, G_gates, Z_gates):
    ''' Cost, gradient and Hessian-vector product of f with JAX autodiff. '''
    grad_f = jax.grad(f, holomorphic=False)
    _, hvp = jax.jvp(lambda g: grad_f(g).conj(), (G_gates,), (Z_gates,))
    return f(G_gates), grad_f(G_gates).conj(), hvp


def random_bra_ket(nsites, nbatch, chi_max, key_ket, key_bra):
    ket0 = tn_helpers.pad_mps_to_max_bonddim(get_random_mps_stacked(key_ket, nsites, nbatch), chi_max)
    bra0 = tn_helpers.pad_mps_to_max_bonddim(get_random_mps_stacked(key_bra, nsites, nbatch), chi_max)
    return ket0, bra0


# ----- Tests -----

def test_get_gates_per_layer():
    print('******  Test get_gates_per_layer *****')
    nsites = 6

    # ------ Create a random brickwall circuit ------
    nlayers = 5
    layers, qubits_in_layer = get_random_layers(nsites, nlayers, first_layer_odd=True)
    Gs = jnp.asarray(util.flatten(layers))
    qubits = util.flatten(qubits_in_layer)
    ind = brickwall_circuit.map_list_of_gates_to_layers(qubits)
    qubits = qubits.tolist()
    gates_per_layer_correct = all(jnp.allclose(Gs[a_], b_) for a, b in zip(ind, layers) for a_, b_ in zip(a, b))
    qubits_per_layer_correct = all(qubits[a_] == b_ for a, b in zip(ind, qubits_in_layer) for a_, b_ in zip(a, b))

    print("\tMatching the gates per layer is correct: ", gates_per_layer_correct)
    print("\tMatching the qubits per layer is correct: ", qubits_per_layer_correct)
    assert gates_per_layer_correct
    assert qubits_per_layer_correct


@TI_cases
def test_forward_pass_first_order(is_TI):
    print(f'****** test_forward_pass_first_order (is_TI={is_TI}) *****')
    # This is not a batched function yet!
    nsites, nlayers, chi_max = 4, 3, 16

    Gs, qubits, qubits_in_layer = random_circuit(nsites, nlayers, is_TI)
    layout = brickwall_circuit.get_circuit_layout(qubits)

    ket0 = get_random_mps(keys[15], nsites)
    ket0_padded = tn_helpers.pad_mps_to_max_bonddim(ket0, chi_max)

    # ------ FoR implementation ------
    psis = brickwall_passes.forward_pass_first_order_cached(
        ket0_padded, Gs, layout.qubits, layout.flat_idxs_fw, layout.layer_ends_fw,
        first_layer_increasing=layout.first_layer_increasing, truncation_dim=chi_max,
        rel_tol=rel_tol, abs_tol=abs_tol, is_TI=is_TI
        )
    psi_final = brickwall_passes.forward_pass_first_order(
        ket0_padded, Gs, layout.qubits, layout.flat_idxs_fw, layout.layer_ends_fw,
        first_layer_increasing=layout.first_layer_increasing, truncation_dim=chi_max,
        rel_tol=rel_tol, abs_tol=abs_tol, is_TI=is_TI
        )

    # ------ Explicit forward intermediate states ------
    res = apply_circuit_dense(util.get_matrices(Gs), qubits_in_layer, get_vector(ket0), is_TI)

    final_states_are_the_same = jnp.allclose(get_vector(psi_final), res[-1])
    forward_intermediate_states_correct = all([jnp.allclose(psi, get_vector(phi)) for psi, phi in zip(res, psis)])

    print("\tFinal states are the same: ", final_states_are_the_same)
    print("\tForward intermediate state are correct: ", forward_intermediate_states_correct)
    assert final_states_are_the_same
    assert forward_intermediate_states_correct


@TI_cases
def test_forward_pass_second_order(is_TI):
    print(f'****** test_forward_pass_second_order (is_TI={is_TI}) *****')
    # This function is not batched yet!
    nsites, nlayers, chi_max = 6, 2, 16

    Gs, Zs, qubits, qubits_in_layer = random_circuit(nsites, nlayers, is_TI, with_direction=True)
    layout = brickwall_circuit.get_circuit_layout(qubits)

    ket0 = get_random_mps(keys[15], nsites)
    ket0_padded = tn_helpers.pad_mps_to_max_bonddim(ket0, chi_max)

    # ------ FoR implementation ------
    psis, Bs, _ = brickwall_passes.forward_pass_second_order(
        ket0_padded, Gs, Zs, layout.qubits, layout.flat_idxs_fw, layout.layer_ends_fw,
        first_layer_increasing=layout.first_layer_increasing, truncation_dim=chi_max,
        rel_tol=rel_tol, abs_tol=abs_tol, is_TI=is_TI
        )

    # ------ Explicit forward intermediate states ------
    res = apply_circuit_dense(util.get_matrices(Gs), qubits_in_layer, get_vector(ket0), is_TI)
    forward_intermediate_states_correct = all([jnp.allclose(psi, get_vector(phi)) for psi, phi in zip(res, psis)])

    print("\tForward intermediate state are correct: ", forward_intermediate_states_correct)
    assert forward_intermediate_states_correct

    # ------ Finite differences ------
    eps = 1e-7
    Gs_eps = Gs + eps*Zs
    if is_TI:
        psi_b_eps = tn_methods.compute_all_intermediate_states_TI(
            Gs_eps, qubits_in_layer, ket0_padded, increasing_merging_order=True,
            truncation_dim=chi_max, rel_tol=rel_tol, abs_tol=abs_tol)[1:]
    else:
        # Regroup the flat gates into layers
        n_per_layer = [len(layer) for layer in qubits_in_layer]
        starts = [sum(n_per_layer[:ell]) for ell in range(len(n_per_layer))]
        layers_eps = [list(Gs_eps[s:s+n]) for s, n in zip(starts, n_per_layer)]
        psi_b_eps = tn_methods.compute_all_intermediate_states(
            layers_eps, qubits_in_layer, ket0_padded, increasing_merging_order=True,
            truncation_dim=chi_max, rel_tol=rel_tol, abs_tol=abs_tol)[1:]
    fd_Dpsi = [(get_vector(_psi_b_eps) - get_vector(_psi_b0)) / eps for _psi_b_eps, _psi_b0 in zip(psi_b_eps, psis[1:])]
    Dpsis = [get_tangent_vector(A, B) for A, B in zip(psis[1:], Bs[1:])]
    FD_test_correct = all([jnp.allclose(a, b) for a, b in zip(Dpsis, fd_Dpsi)])

    print("\tFD test for directional derivative of intermediate states correct: ", FD_test_correct)
    assert FD_test_correct


@TI_cases
def test_compute_loss_gradient_F(is_TI):
    print(f"****** test_compute_loss_gradient_F (is_TI={is_TI}) ******")
    nsites, nlayers, nbatch, chi_max = 6, 3, 5, 16

    Gs, qubits, qubits_in_layer = random_circuit(nsites, nlayers, is_TI)
    ket0_padded, bra0_padded = random_bra_ket(nsites, nbatch, chi_max, keys[13], keys[14])

    # ------ JAX ------
    f = dense_cost('F', ket0_padded, bra0_padded, qubits_in_layer, is_TI)
    G_gates = util.get_matrices(Gs)
    cost_jax = f(G_gates)
    grad_jax = jax.grad(f, holomorphic=False)(G_gates).conj()

    # ------ TN methods (FoR) ------
    cost_F, grad_F = objective_function.compute_loss_gradient_F(
        ket0_padded, bra0_padded, Gs, brickwall_circuit.get_circuit_layout(qubits),
        truncation_dim=chi_max, rel_tol=rel_tol, abs_tol=abs_tol, is_TI=is_TI)

    cost_correct = jnp.allclose(cost_F, cost_jax)
    gradient_correct = jnp.allclose(util.get_matrices(grad_F), grad_jax)

    print("\tCost correct: ", cost_correct)
    print("\tGradient correct: ", gradient_correct)
    assert cost_correct
    assert gradient_correct


@TI_cases
def test_compute_loss_gradient_hvp_F(is_TI):
    print(f"****** test_compute_loss_gradient_hvp_F (is_TI={is_TI}) ******")
    nsites, nlayers, nbatch, chi_max = 6, 3, 5, 16

    Gs, Zs, qubits, qubits_in_layer = random_circuit(nsites, nlayers, is_TI, with_direction=True)
    ket0_padded, bra0_padded = random_bra_ket(nsites, nbatch, chi_max, keys[13], keys[14])

    # ------ JAX ------
    f = dense_cost('F', ket0_padded, bra0_padded, qubits_in_layer, is_TI)
    cost_jax, grad_jax, hvp_jax = dense_cost_gradient_hvp(f, util.get_matrices(Gs), util.get_matrices(Zs))

    # ------ TN methods (FoR) ------
    cost_F, grad_F, hvp_F = objective_function.compute_loss_gradient_hvp_F(
        ket0_padded, bra0_padded, Gs, Zs, brickwall_circuit.get_circuit_layout(qubits),
        truncation_dim=chi_max, rel_tol=rel_tol, abs_tol=abs_tol, is_TI=is_TI
        )

    cost_correct = jnp.allclose(cost_F, cost_jax)
    gradient_correct = jnp.allclose(util.get_matrices(grad_F), grad_jax)
    hvp_correct = jnp.allclose(util.get_matrices(hvp_F), hvp_jax)

    print("\tCost correct: ", cost_correct)
    print("\tGradient correct: ", gradient_correct)
    print("\tHVP correct: ", hvp_correct)
    assert cost_correct
    assert gradient_correct
    assert hvp_correct


@TI_cases
def test_compute_loss_HST(is_TI):
    print(f"****** test_compute_loss_HST (is_TI={is_TI}) ******")
    nsites, nlayers, nbatch, chi_max = 6, 3, 5, 16

    Gs, qubits, qubits_in_layer = random_circuit(nsites, nlayers, is_TI)
    ket0_padded, bra0_padded = random_bra_ket(nsites, nbatch, chi_max, keys[15], keys[16])

    # ------ JAX ------
    f = dense_cost('HST', ket0_padded, bra0_padded, qubits_in_layer, is_TI)
    cost_jax = f(util.get_matrices(Gs))

    # ------ FoR ------
    cost_HS = objective_function.compute_loss_HST(
        ket0_padded, bra0_padded, Gs, brickwall_circuit.get_circuit_layout(qubits),
        truncation_dim=chi_max, rel_tol=rel_tol, abs_tol=abs_tol, is_TI=is_TI)

    cost_correct = jnp.allclose(cost_HS, cost_jax)

    print("\tCost correct: ", cost_correct)
    assert cost_correct


@TI_cases
def test_compute_loss_gradient_HST(is_TI):
    print(f"****** test_compute_loss_gradient_HST (is_TI={is_TI}) ******")
    nsites, nlayers, nbatch, chi_max = 6, 3, 5, 16

    Gs, qubits, qubits_in_layer = random_circuit(nsites, nlayers, is_TI)
    ket0_padded, bra0_padded = random_bra_ket(nsites, nbatch, chi_max, keys[15], keys[16])

    # ------ JAX ------
    f = dense_cost('HST', ket0_padded, bra0_padded, qubits_in_layer, is_TI)
    G_gates = util.get_matrices(Gs)
    cost_jax = f(G_gates)
    grad_jax = jax.grad(f, holomorphic=False)(G_gates).conj()

    # ------ FoR ------
    cost_HS, grad_HS = objective_function.compute_loss_gradient_HST(
        ket0_padded, bra0_padded, Gs, brickwall_circuit.get_circuit_layout(qubits),
        truncation_dim=chi_max, rel_tol=rel_tol, abs_tol=abs_tol, is_TI=is_TI)

    cost_correct = jnp.allclose(cost_HS, cost_jax)
    gradient_correct = jnp.allclose(util.get_matrices(grad_HS), grad_jax)

    print("\tCost correct: ", cost_correct)
    print("\tGradient correct: ", gradient_correct)
    assert cost_correct
    assert gradient_correct


@TI_cases
def test_compute_loss_gradient_hvp_HST(is_TI):
    print(f"****** test_compute_loss_gradient_hvp_HST (is_TI={is_TI}) ******")
    nsites, nlayers, nbatch, chi_max = 6, 3, 5, 16

    Gs, Zs, qubits, qubits_in_layer = random_circuit(nsites, nlayers, is_TI, with_direction=True)
    ket0_padded, bra0_padded = random_bra_ket(nsites, nbatch, chi_max, keys[15], keys[16])

    # ------ JAX ------
    f = dense_cost('HST', ket0_padded, bra0_padded, qubits_in_layer, is_TI)
    cost_jax, grad_jax, hvp_jax = dense_cost_gradient_hvp(f, util.get_matrices(Gs), util.get_matrices(Zs))

    # ------ FoR ------
    cost_HS, grad_HS, hvp_HS = objective_function.compute_loss_gradient_hvp_HST(
        ket0_padded, bra0_padded, Gs, Zs, brickwall_circuit.get_circuit_layout(qubits),
        truncation_dim=chi_max, rel_tol=rel_tol, abs_tol=abs_tol, is_TI=is_TI)

    cost_correct = jnp.allclose(cost_HS, cost_jax)
    gradient_correct = jnp.allclose(util.get_matrices(grad_HS), grad_jax)
    hvp_correct = jnp.allclose(util.get_matrices(hvp_HS), hvp_jax)

    print("\tCost correct: ", cost_correct)
    print("\tGradient correct: ", gradient_correct)
    print("\tHVP correct: ", hvp_correct)
    assert cost_correct
    assert gradient_correct
    assert hvp_correct


@TI_cases
def test_compute_loss_gradient_hvp_HST_all_functions(is_TI):
    print(f"****** test_compute_loss_gradient_hvp_HST_all_functions (is_TI={is_TI}) ******")
    nsites, nlayers, nbatch, chi_max = 6, 3, 5, 16

    Gs, Zs, qubits, qubits_in_layer = random_circuit(nsites, nlayers, is_TI, with_direction=True)
    layout = brickwall_circuit.get_circuit_layout(qubits)
    ket0_padded, bra0_padded = random_bra_ket(nsites, nbatch, chi_max, keys[15], keys[16])

    # ------ JAX ------
    f = dense_cost('HST', ket0_padded, bra0_padded, qubits_in_layer, is_TI)
    cost_jax, grad_jax, hvp_jax = dense_cost_gradient_hvp(f, util.get_matrices(Gs), util.get_matrices(Zs))

    # ------ loss ------
    cost_HS1 = objective_function.compute_loss_HST(
        ket0_padded, bra0_padded, Gs, layout, chi_max, rel_tol, abs_tol, is_TI=is_TI)

    cost_correct = jnp.allclose(cost_HS1, cost_jax)
    print("\tCost correct: ", cost_correct)
    assert cost_correct

    # ------ loss, gradient ------
    cost_HS2, grad_HS2 = objective_function.compute_loss_gradient_HST(
        ket0_padded, bra0_padded, Gs, layout, chi_max, rel_tol, abs_tol, is_TI=is_TI)

    cost_correct = jnp.allclose(cost_HS2, cost_jax)
    gradient_correct = jnp.allclose(util.get_matrices(grad_HS2), grad_jax)
    print("\tCost correct: ", cost_correct)
    print("\tGradient correct: ", gradient_correct)
    assert cost_correct
    assert gradient_correct

    # ------ loss, gradient, hvp ------
    cost_HS3, grad_HS3, hvp_HS3 = objective_function.compute_loss_gradient_hvp_HST(
        ket0_padded, bra0_padded, Gs, Zs, layout,
        truncation_dim=chi_max, rel_tol=rel_tol, abs_tol=abs_tol, is_TI=is_TI)

    cost_correct = jnp.allclose(cost_HS3, cost_jax)
    gradient_correct = jnp.allclose(util.get_matrices(grad_HS3), grad_jax)
    hvp_correct = jnp.allclose(util.get_matrices(hvp_HS3), hvp_jax)

    print("\tCost correct: ", cost_correct)
    print("\tGradient correct: ", gradient_correct)
    print("\tHVP correct: ", hvp_correct)
    assert cost_correct
    assert gradient_correct
    assert hvp_correct

    print("All costs are the same: ", cost_HS1, cost_HS2, cost_HS3)


@TI_cases
def test_compute_riemannian_loss_gradient_hvp_HST_all_functions(is_TI):
    print(f"****** test_compute_riemannian_loss_gradient_hvp_HST_all_functions (is_TI={is_TI}) ******")
    nsites, nlayers, nbatch, chi_max = 6, 3, 5, 16

    Gs, Zs, qubits, _ = random_circuit(nsites, nlayers, is_TI, with_direction=True)
    Zs = riemannian_manifold.project_to_tangent_space(Gs, Zs)
    layout = brickwall_circuit.get_circuit_layout(qubits)
    ket0_padded, bra0_padded = random_bra_ket(nsites, nbatch, chi_max, keys[15], keys[16])

    # ------ loss ------
    cost_HS1 = objective_function.compute_loss_HST(
        ket0_padded, bra0_padded, Gs, layout, chi_max, rel_tol, abs_tol, is_TI=is_TI)

    # ------ loss, gradient ------
    cost_HS2, grad_HS2 = objective_function.compute_riemannian_loss_gradient_HST(
        ket0_padded, bra0_padded, Gs, layout, chi_max, rel_tol, abs_tol, is_TI=is_TI)

    # ------ loss, gradient, hvp ------
    cost_HS3, grad_HS3, hvp_HS3 = objective_function.compute_riemannian_loss_gradient_hvp_HST(
        ket0_padded, bra0_padded, Gs, Zs, layout, chi_max, rel_tol, abs_tol, is_TI=is_TI)

    # ------ hvp ------
    hvp_HS4 = objective_function.compute_riemannian_hvp_HST(
        ket0_padded, bra0_padded, Gs, Zs, layout, chi_max, rel_tol, abs_tol, is_TI=is_TI)

    gradients_the_same = jnp.allclose(grad_HS2, grad_HS3)
    hvps_the_same = jnp.allclose(hvp_HS3, hvp_HS4)

    print("All costs are the same: ", cost_HS1, cost_HS2, cost_HS3)
    print("All gradients are the same: ", gradients_the_same)
    print("All HVPs are the same: ", hvps_the_same)
    assert gradients_the_same
    assert hvps_the_same

    # ------ Test looping versions ------
    cost_loop = objective_function.compute_loss_HST_loop(
        ket0_padded, bra0_padded, Gs, layout, chi_max, rel_tol, abs_tol, is_TI=is_TI)
    cost_loop.block_until_ready()
    costs_match = jnp.allclose(cost_HS1, cost_loop)

    print("Looped cost matches: ", costs_match)
    assert costs_match

    cost_grad_loop, grad_loop = objective_function.compute_riemannian_loss_gradient_HST_loop(
        ket0_padded, bra0_padded, Gs, layout, chi_max, rel_tol, abs_tol, is_TI=is_TI)
    cost_grad_loop.block_until_ready()
    costs_match = jnp.allclose(cost_HS2, cost_grad_loop)
    grads_match = jnp.allclose(grad_HS2, grad_loop)

    print("Looped cost+grad matches cost: ", costs_match)
    print("Looped cost+grad matches grad: ", grads_match)
    assert costs_match
    assert grads_match

    cost_hvp_loop, grad_hvp_loop, hvp_loop = objective_function.compute_riemannian_loss_gradient_hvp_HST_loop(
        ket0_padded, bra0_padded, Gs, Zs, layout, chi_max, rel_tol, abs_tol, is_TI=is_TI)
    cost_hvp_loop.block_until_ready()
    costs_match = jnp.allclose(cost_HS3, cost_hvp_loop)
    grads_match = jnp.allclose(grad_HS3, grad_hvp_loop)
    hvps_match = jnp.allclose(hvp_HS3, hvp_loop)
    print("Looped cost+grad+hvp matches cost: ", costs_match)
    print("Looped cost+grad+hvp matches grad: ", grads_match)
    print("Looped cost+grad+hvp matches hvp: ", hvps_match)
    assert costs_match
    assert grads_match
    assert hvps_match


@TI_cases
def test_riemannian_projection(is_TI):
    print(f"****** test_riemannian_projection (is_TI={is_TI}) ******")
    nsites, nlayers, nbatch, chi_max = 6, 3, 5, 16

    Gs, Zs, qubits, qubits_in_layer = random_circuit(nsites, nlayers, is_TI, with_direction=True)
    Zs = riemannian_manifold.project_to_tangent_space(Gs, Zs)
    G_gates = util.get_matrices(Gs)
    Z_gates = util.get_matrices(Zs)
    ket0_padded, bra0_padded = random_bra_ket(nsites, nbatch, chi_max, keys[15], keys[16])

    # ------ JAX ------
    f = dense_cost('HST', ket0_padded, bra0_padded, qubits_in_layer, is_TI)
    grad_f_HS = jax.grad(f, holomorphic=False)

    def riem_grad(g):
        return riemannian_manifold.project_to_tangent_space(
            util.get_tensors(g), util.get_tensors(grad_f_HS(util.get_matrices(g)).conj())
            ).reshape((-1, 4, 4))

    def hvp_fd(G, xi, grad_R, eps=1e-5):
        g0 = grad_R(G)  # gradient at base point
        G_eps = riemannian_manifold.retract_to_manifold(G, eps*xi)
        g_eps = grad_R(G_eps)  # gradient at perturbed point
        # Difference quotient, projected back to the tangent space at G
        res = (g_eps - g0) / eps
        return riemannian_manifold.project_to_tangent_space(util.get_tensors(G), util.get_tensors(res))

    hvp_FD = hvp_fd(G_gates, Z_gates, riem_grad, eps=1e-7).reshape((-1, 4, 4))

    # ------ FoR ------
    cost_HS, grad_HS, hvp_HS = objective_function.compute_riemannian_loss_gradient_hvp_HST(
        ket0_padded, bra0_padded, Gs, Zs, brickwall_circuit.get_circuit_layout(qubits), truncation_dim=chi_max,
        rel_tol=rel_tol, abs_tol=abs_tol, is_TI=is_TI
        )

    hvp_correct = jnp.allclose(util.get_matrices(hvp_HS), hvp_FD)

    print("\tHVP correct: ", hvp_correct)
    assert hvp_correct


def main():
    test_get_gates_per_layer()
    print()
    for is_TI in [False, True]:
        test_forward_pass_first_order(is_TI)
        print()
        test_forward_pass_second_order(is_TI)
        print()
        test_compute_loss_gradient_F(is_TI)
        print()
        test_compute_loss_HST(is_TI)
        print()
        test_compute_loss_gradient_HST(is_TI)
        print()
        test_compute_loss_gradient_hvp_F(is_TI)
        print()
        test_compute_loss_gradient_hvp_HST(is_TI)
        print()
        test_riemannian_projection(is_TI)
        print()
        test_compute_loss_gradient_hvp_HST_all_functions(is_TI)
        print()
        test_compute_riemannian_loss_gradient_hvp_HST_all_functions(is_TI)
        print()


if __name__ == "__main__":
    main()
