import os 

# This needs to be set before importing jax!
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=5"

import jax.numpy as jnp
import jax
jax.config.update('jax_platform_name', 'cpu')
from jax import random
from rqcopt_mps import *


key = random.PRNGKey(0)
keys = random.split(key, 20)


def test_get_gates_per_layer():
    print('******  Test get_gates_per_layer *****')
    nsites = 6

    # ------- Create a random brickwall circuit -------
    nlayers = 5
    layers, qubits_in_layer = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Gs = jnp.asarray(util.flatten(layers))
    qubits = util.flatten(qubits_in_layer)
    ind = brickwall_circuit.map_list_of_gates_to_layers(qubits)
    qubits = qubits.tolist()
    gates_per_layer_correct = all(jnp.allclose(Gs[a_], b_) for a,b in zip(ind, layers) for a_,b_ in zip(a,b))
    qubits_per_layer_correct = all(qubits[a_] == b_ for a,b in zip(ind, qubits_in_layer) for a_,b_ in zip(a,b))

    print("\tMatching the gates per layer is correct: ", gates_per_layer_correct)
    print("\tMatching the qubits per layer is correct: ", qubits_per_layer_correct)
    assert gates_per_layer_correct
    assert qubits_per_layer_correct


def test_forward_pass_first_order():
    print('****** test_forward_pass_first_order *****')
    # This is not a batched function yet!

    nsites, I, I2, chi_max = 4, jnp.eye(2), jnp.eye(4), 16

    # ------- Create a random brickwall circuit -------
    nlayers = 3
    layers, qubits_in_layer = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Gs = jnp.asarray(util.flatten(layers))
    qubits = util.flatten(qubits_in_layer)

    # ------- Non-jittable preprocessing -------
    idx_per_layer_fw = brickwall_circuit.map_list_of_gates_to_layers(qubits)  # indices for gates 2..n
    flat_idxs_fw = util.flatten(idx_per_layer_fw)
    layer_ends_fw = brickwall_circuit.gate_is_at_boundary(idx_per_layer_fw)

    # ------- Create random initial ket -------
    ket0 = mps.get_random_mps(keys[15], nsites)
    ket0_padded = tn_helpers.pad_mps_to_max_bonddim(ket0, chi_max)

    # ------- FoR implementation -------
    psis = brickwall_passes.forward_pass_first_order_cached(
        ket0_padded, Gs, qubits, flat_idxs_fw, layer_ends_fw, 
        first_layer_increasing=True, truncation_dim=chi_max,
        rel_tol=1e-11, abs_tol=1e-14
        )

    psi_final = brickwall_passes.forward_pass_first_order(
        ket0_padded, Gs, qubits, flat_idxs_fw, layer_ends_fw, 
        first_layer_increasing=True, truncation_dim=chi_max,
        rel_tol=1e-11, abs_tol=1e-14
        )
    

    # ------- Explicit implementation of forward intermediate states -------
    psi_0 = mps.get_vector(ket0)
    def get_exact_result(psi_0):
        psi_1 = util.tensor_product([layers[0][0].reshape((4,4)),I2]) @ psi_0
        psi_2 = util.tensor_product([I2, layers[0][1].reshape((4,4))]) @ psi_1
        psi_3 = util.tensor_product([I,layers[1][0].reshape((4,4)),I]) @ psi_2 
        psi_4 = util.tensor_product([layers[-1][0].reshape((4,4)),I2]) @ psi_3
        psi_5 = util.tensor_product([I2]+[layers[-1][-1].reshape((4,4))]) @ psi_4
        res = [psi_5, psi_4, psi_3, psi_2, psi_1, psi_0][::-1]
        return res
    
    res = get_exact_result(psi_0)

    losses_are_the_same = jnp.allclose(mps.get_vector(psi_final), res[-1])
    forward_intermediate_states_correct = all([jnp.allclose(psi, mps.get_vector(phi)) for psi, phi in zip(res, psis)])

    print("\tLosses are the same: ", )
    print("\tForward intermediate state are correct: ", forward_intermediate_states_correct)
    assert losses_are_the_same
    assert forward_intermediate_states_correct


def test_forward_pass_second_order():
    print('****** test_forward_pass_second_order *****')
    # This function is not batched yet!

    nsites, I, I2, chi_max = 6, jnp.eye(2), jnp.eye(4), 16

    # ------- Create a random brickwall circuit -------
    nlayers = 2
    G_layers, qubits_in_layer = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Z_layers, _ = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Gs = jnp.asarray(util.flatten(G_layers))
    Zs = jnp.asarray(util.flatten(Z_layers))
    qubits = util.flatten(qubits_in_layer)
    idx_per_layer_fw = brickwall_circuit.map_list_of_gates_to_layers(qubits)
    flat_idxs_fw = util.flatten(idx_per_layer_fw)
    layer_ends_fw = brickwall_circuit.gate_is_at_boundary(idx_per_layer_fw)

    # ------- Create random initial ket -------
    ket0 = mps.get_random_mps(keys[15], nsites)
    ket0_padded = tn_helpers.pad_mps_to_max_bonddim(ket0, chi_max)

    # ------- FoR implementation -------
    psis, Dpsis = brickwall_passes.forward_pass_second_order(
        ket0_padded, Gs, Zs, qubits, flat_idxs_fw, layer_ends_fw, 
        first_layer_increasing=True, truncation_dim=chi_max,
        rel_tol=1e-11, abs_tol=1e-14
        )

    # ------- Explicit implementation of forward intermediate states -------
    psi_0 = mps.get_vector(ket0)
    def get_exact_result(psi_0):
        psi_1 = util.tensor_product([G_layers[0][0].reshape((4,4)),I2,I2]) @ psi_0
        psi_2 = util.tensor_product([I2,G_layers[0][1].reshape((4,4)),I2]) @ psi_1
        psi_3 = util.tensor_product([I2,I2,G_layers[0][-1].reshape((4,4))]) @ psi_2 
        psi_4 = util.tensor_product([I,I2]+[G_layers[-1][0].reshape((4,4))]+[I]) @ psi_3
        psi_5 = util.tensor_product([I]+[G_layers[-1][-1].reshape((4,4))] + [I2, I]) @ psi_4
        res = [psi_5, psi_4, psi_3, psi_2, psi_1, psi_0][::-1]
        return res
    
    res = get_exact_result(psi_0)
    forward_intermediate_states_correct = all([jnp.allclose(psi, mps.get_vector(phi)) for psi, phi in zip(res, psis)])

    print("\tForward intermediate state are correct: ", forward_intermediate_states_correct)
    assert forward_intermediate_states_correct

    # ------- Finite differences -------
    eps = 1e-7
    psi_b0 = psis[1:]
    layers_eps = [[G + eps*Z for G,Z in zip(_G_layers, _Z_layers)] for _G_layers,_Z_layers in zip(G_layers, Z_layers)]
    psi_b_eps = tn_methods.compute_all_intermediate_states(
        layers_eps, qubits_in_layer, ket0_padded, increasing_merging_order=True, truncation_dim=chi_max,rel_tol=1e-11, abs_tol=1e-14)[1:]
    fd_Dpsi = [(mps.get_vector(_psi_b_eps) - mps.get_vector(_psi_b0)) / eps for _psi_b_eps,_psi_b0 in zip(psi_b_eps,psi_b0)]
    FD_test_correct = all([jnp.allclose(mps.get_vector(a),b) for a,b in zip(Dpsis[1:-1], fd_Dpsi)])
    
    print("\tFD test for directional derivative of intermediate states correct: ", FD_test_correct)
    assert FD_test_correct
    

def test_compute_loss_gradient_F():
    print("****** test_compute_loss_gradient_F ******")
    nsites, nbatch, I, chi_max = 6, 5, jnp.eye(2), 16

    # ------- Create a random brickwall circuit + direction -------
    nlayers = 3
    layers, qubits_in_layer = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Gs = jnp.asarray(util.flatten(layers))
    qubits = util.flatten(qubits_in_layer)

    # ------- Create random bra and ket environments -------
    ket0 = mps.get_random_mps_stacked(keys[13], nsites, nbatch)
    ket0_padded = tn_helpers.pad_mps_to_max_bonddim(ket0, chi_max)
    bra0 = mps.get_random_mps_stacked(keys[14], nsites, nbatch)
    bra0_padded = tn_helpers.pad_mps_to_max_bonddim(bra0, chi_max)


    # ------- JAX -------
    def f(gates):  # Frobenius
        res = 0
        for _ket0, _bra0 in zip(ket0_padded.copy(), bra0_padded.copy()):
            ket1 = util.tensor_product(list(gates[:3])) @ mps.get_vector(_ket0)
            ket2 = util.tensor_product([I]+list(reversed(gates[3:5]))+[I]) @ ket1
            ket3 = util.tensor_product(list(gates[5::])) @ ket2
            res += jnp.dot(mps.get_vector(_bra0), ket3).real
        return 1 - res/nbatch
    grad_f = jax.grad(f, holomorphic=False)
    cost_jax = f(brickwall_circuit.get_matrices(Gs))
    grad_jax = grad_f(brickwall_circuit.get_matrices(Gs)).conj()

    # ------- TN methods (FoR) -------
    cost_F, grad_F = objective_function.compute_loss_gradient_F(
        ket0_padded, bra0_padded, Gs, qubits, first_layer_increasing=True, 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14)
        
    cost_correct = jnp.allclose(cost_F, cost_jax)
    gradient_correct = jnp.allclose(brickwall_circuit.get_matrices(grad_F), grad_jax)

    print("\tCost correct: ", cost_correct)
    print("\tGradient correct: ", gradient_correct)
    assert cost_correct
    assert gradient_correct


def test_compute_loss_gradient_hvp_F():
    print("****** compute_loss_gradient_hvp_F ******")
    nsites, I, nbatch, chi_max = 6, jnp.eye(2), 5, 16

    # ------- Create a random brickwall circuit + direction -------
    nlayers = 3
    G_layers, qubits_in_layer = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Z_layers, _ = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Gs = jnp.asarray(util.flatten(G_layers))
    Zs = jnp.asarray(util.flatten(Z_layers))
    qubits = util.flatten(qubits_in_layer)
    G_gates = brickwall_circuit.get_matrices(Gs)
    Z_gates = brickwall_circuit.get_matrices(Zs)

    # ------- Create random bra and ket environments -------
    ket0 = mps.get_random_mps_stacked(keys[13], nsites, nbatch)
    ket0_padded = tn_helpers.pad_mps_to_max_bonddim(ket0, chi_max)
    bra0 = mps.get_random_mps_stacked(keys[14], nsites, nbatch)
    bra0_padded = tn_helpers.pad_mps_to_max_bonddim(bra0, chi_max)

    # ------- JAX -------
    def f(gates):  # Frobenius
        res = 0
        for _ket0, _bra0 in zip(ket0_padded, bra0_padded):
            ket1 = util.tensor_product(list(gates[:3])) @ mps.get_vector(_ket0)
            ket2 = util.tensor_product([I]+list(reversed(gates[3:5]))+[I]) @ ket1
            ket3 = util.tensor_product(list(gates[5::])) @ ket2
            res += jnp.dot(mps.get_vector(_bra0), ket3).real
        return 1 - res/nbatch
    grad_f = jax.grad(f, holomorphic=False)

    def hvp_real_overlap_FoR(G_gates, Z_gates):  
        _, hvp = jax.jvp(lambda g: grad_f(g).conj(), (G_gates,), (Z_gates,))
        return hvp
    cost_jax = f(G_gates)
    grad_jax = grad_f(G_gates).conj()
    hvp_jax = hvp_real_overlap_FoR(G_gates, Z_gates)


    # ------- TN methods (FoR) -------
    cost_F, grad_F, hvp_F = objective_function.compute_loss_gradient_hvp_F(
        ket0_padded, bra0_padded, Gs, Zs, qubits, first_layer_increasing=True, 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14)
        
    cost_correct = jnp.allclose(cost_F, cost_jax)
    gradient_correct = jnp.allclose(grad_F.reshape((-1,4,4)), grad_jax)
    hvp_correct = jnp.allclose(hvp_F.reshape((-1,4,4)), hvp_jax)
        
    print("\tCost correct: ", cost_correct)
    print("\tGradient correct: ", gradient_correct)
    print("\tHVP correct: ", hvp_correct)
    assert cost_correct
    assert gradient_correct
    assert hvp_correct


def test_compute_loss_HST():
    print("****** test_compute_loss_HST ******")
    nsites, I, nbatch, chi_max = 6, jnp.eye(2), 5, 16

    # ------- Create a random brickwall circuit -------
    nlayers = 3
    layers, qubits_in_layer = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Gs = jnp.asarray(util.flatten(layers))
    qubits = util.flatten(qubits_in_layer)
    G_gates = brickwall_circuit.get_matrices(Gs)

    # ------- Create random bra and ket environments -------
    ket0 = mps.get_random_mps_stacked(keys[15], nsites, nbatch)
    bra0 = mps.get_random_mps_stacked(keys[16], nsites, nbatch)
    ket0_padded = tn_helpers.pad_mps_to_max_bonddim(ket0, chi_max)
    bra0_padded = tn_helpers.pad_mps_to_max_bonddim(bra0, chi_max)
    
    # ------- JAX -------
    def f(gates):  # Overlap
        res = 0
        for _ket0, _bra0 in zip(ket0_padded, bra0_padded):
            ket1 = util.tensor_product(list(gates[:3])) @ mps.get_vector(_ket0)
            ket2 = util.tensor_product([I]+list(reversed(gates[3:5]))+[I]) @ ket1
            ket3 = util.tensor_product(list(gates[5::])) @ ket2
            res += jnp.abs(jnp.dot(mps.get_vector(_bra0), ket3))**2
            #res += jnp.dot(mps.get_vector(_bra0), ket3)
        return 1 - res/nbatch
    cost_jax = f(G_gates)

    # ------- FoR -------
    cost_HS = objective_function.compute_loss_HST(
        ket0_padded, bra0_padded, Gs, qubits, first_layer_increasing=True, 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14, is_TI=False
        )
    
    cost_correct = jnp.allclose(cost_HS, cost_jax)

    print("\tCost correct: ", cost_correct)
    assert cost_correct


def test_compute_loss_gradient_HST():
    print("****** test_compute_loss_gradient_HST ******")
    nsites, I, nbatch, chi_max = 6, jnp.eye(2), 5, 16

    # ------- Create a random brickwall circuit -------
    nlayers = 3
    layers, qubits_in_layer = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Gs = jnp.asarray(util.flatten(layers))
    qubits = util.flatten(qubits_in_layer)
    G_gates = brickwall_circuit.get_matrices(Gs)

    # ------- Create random bra and ket environments -------
    ket0 = mps.get_random_mps_stacked(keys[15], nsites, nbatch)
    bra0 = mps.get_random_mps_stacked(keys[16], nsites, nbatch)
    ket0_padded = tn_helpers.pad_mps_to_max_bonddim(ket0, chi_max)
    bra0_padded = tn_helpers.pad_mps_to_max_bonddim(bra0, chi_max)
    
    # ------- JAX -------
    def f(gates):  # Overlap
        res = 0
        for _ket0, _bra0 in zip(ket0_padded, bra0_padded):
            ket1 = util.tensor_product(list(gates[:3])) @ mps.get_vector(_ket0)
            ket2 = util.tensor_product([I]+list(reversed(gates[3:5]))+[I]) @ ket1
            ket3 = util.tensor_product(list(gates[5::])) @ ket2
            res += jnp.abs(jnp.dot(mps.get_vector(_bra0), ket3))**2
        return 1 - res/nbatch
    grad_f_HS = jax.grad(f, holomorphic=False)
    cost_jax = f(G_gates)
    grad_jax = grad_f_HS(G_gates).conj()

    # ------- FoR -------
    cost_HS, grad_HS = objective_function.compute_loss_gradient_HST(
        ket0_padded, bra0_padded, Gs, qubits, first_layer_increasing=True, 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14)
    
    cost_correct = jnp.allclose(cost_HS, cost_jax)
    gradient_correct = jnp.allclose(grad_HS.reshape((-1,4,4)), grad_jax)

    print("\tCost correct: ", cost_correct)
    print("\tGradient correct: ", gradient_correct)
    assert cost_correct
    assert gradient_correct


def test_compute_loss_gradient_hvp_HST():
    print("****** test_compute_loss_gradient_hvp_HST ******")
    nsites, I, nbatch, chi_max = 6, jnp.eye(2), 5, 16

    # ------- Create a random brickwall circuit -------
    nlayers = 3
    G_layers, qubits_in_layer = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Z_layers, _ = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Gs = jnp.asarray(util.flatten(G_layers))
    Zs = jnp.asarray(util.flatten(Z_layers))
    qubits = util.flatten(qubits_in_layer)
    G_gates = brickwall_circuit.get_matrices(Gs)
    Z_gates = brickwall_circuit.get_matrices(Zs)

    # ------- Create random bra and ket environments -------
    ket0 = mps.get_random_mps_stacked(keys[15], nsites, nbatch)
    bra0 = mps.get_random_mps_stacked(keys[16], nsites, nbatch)
    ket0_padded = tn_helpers.pad_mps_to_max_bonddim(ket0, chi_max)
    bra0_padded = tn_helpers.pad_mps_to_max_bonddim(bra0, chi_max)

    # ------- JAX -------
    def f(gates):  # Frobenius
        res = 0
        for _ket0,_bra0 in zip(ket0_padded, bra0_padded):
            ket1 = util.tensor_product(list(gates[:3])) @ mps.get_vector(_ket0)
            ket2 = util.tensor_product([I]+list(reversed(gates[3:5]))+[I]) @ ket1
            ket3 = util.tensor_product(list(gates[5::])) @ ket2
            res += jnp.abs(jnp.dot(mps.get_vector(_bra0), ket3))**2
        return 1 - res/nbatch
    grad_f_HS = jax.grad(f, holomorphic=False)

    def hvp_HS_FoR(G_gates, Z_gates):  
        _, hvp = jax.jvp(lambda g: grad_f_HS(g).conj(), (G_gates,), (Z_gates,))
        return hvp

    cost_jax = f(G_gates)
    grad_jax = grad_f_HS(G_gates).conj()
    hvp_jax = hvp_HS_FoR(G_gates, Z_gates)

    # ------- FoR -------
    cost_HS, grad_HS, hvp_HS = objective_function.compute_loss_gradient_hvp_HST(
        ket0_padded, bra0_padded, Gs, Zs, qubits, first_layer_increasing=True, 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14)
                
    cost_correct = jnp.allclose(cost_HS, cost_jax)
    gradient_correct = jnp.allclose(grad_HS.reshape((-1,4,4)), grad_jax)
    hvp_correct = jnp.allclose(hvp_HS.reshape((-1,4,4)), hvp_jax)

    print("\tCost correct: ", cost_correct)
    print("\tGradient correct: ", gradient_correct)
    print("\tHVP correct: ", hvp_correct)
    assert cost_correct
    assert gradient_correct
    assert hvp_correct


def test_compute_loss_gradient_hvp_HST_all_functions():
    print("****** test_compute_loss_gradient_hvp_HST_all_functions ******")
    nsites, I, nbatch, chi_max = 6, jnp.eye(2), 5, 16

    # ------- Create a random brickwall circuit -------
    nlayers = 3
    G_layers, qubits_in_layer = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Z_layers, _ = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Gs = jnp.asarray(util.flatten(G_layers))
    Zs = jnp.asarray(util.flatten(Z_layers))
    qubits = util.flatten(qubits_in_layer)
    G_gates = brickwall_circuit.get_matrices(Gs)
    Z_gates = brickwall_circuit.get_matrices(Zs)

    # ------- Create random bra and ket environments -------
    ket0 = mps.get_random_mps_stacked(keys[15], nsites, nbatch)
    bra0 = mps.get_random_mps_stacked(keys[16], nsites, nbatch)
    ket0_padded = tn_helpers.pad_mps_to_max_bonddim(ket0, chi_max)
    bra0_padded = tn_helpers.pad_mps_to_max_bonddim(bra0, chi_max)

    # ------- JAX -------
    def f(gates):  # Frobenius
        res = 0
        for _ket0,_bra0 in zip(ket0_padded, bra0_padded):
            ket1 = util.tensor_product(list(gates[:3])) @ mps.get_vector(_ket0)
            ket2 = util.tensor_product([I]+list(reversed(gates[3:5]))+[I]) @ ket1
            ket3 = util.tensor_product(list(gates[5::])) @ ket2
            res += jnp.abs(jnp.dot(mps.get_vector(_bra0), ket3))**2
        return 1 - res/nbatch
    grad_f_HS = jax.grad(f, holomorphic=False)

    def hvp_HS_FoR(G_gates, Z_gates):  
        _, hvp = jax.jvp(lambda g: grad_f_HS(g).conj(), (G_gates,), (Z_gates,))
        return hvp

    cost_jax = f(G_gates)
    grad_jax = grad_f_HS(G_gates).conj()
    hvp_jax = hvp_HS_FoR(G_gates, Z_gates)


    # ------- loss -------
    cost_HS1 = objective_function.compute_loss_HST(
        ket0_padded, bra0_padded, Gs, qubits, True,  chi_max, rel_tol=1e-11, abs_tol=1e-14)
    cost_correct = jnp.allclose(cost_HS1, cost_jax)

    print("\tCost correct: ", cost_correct)
    assert cost_correct

    # ------- loss, gradient -------
    cost_HS2, grad_HS2 = objective_function.compute_loss_gradient_HST(
        ket0_padded, bra0_padded, Gs, qubits, True, chi_max, rel_tol=1e-11, abs_tol=1e-14)
    cost_correct = jnp.allclose(cost_HS2, cost_jax)
    gradient_correct = jnp.allclose(grad_HS2.reshape((-1,4,4)), grad_jax)

    print("\tCost correct: ", cost_correct)
    print("\tGradient correct: ", gradient_correct)
    assert cost_correct
    assert gradient_correct

    # ------ loss, gradient, hvp ---------
    cost_HS3, grad_HS3, hvp_HS3 = objective_function.compute_loss_gradient_hvp_HST(
        ket0_padded, bra0_padded, Gs, Zs, qubits, first_layer_increasing=True, 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14
        )
    cost_correct = jnp.allclose(cost_HS3, cost_jax)
    gradient_correct = jnp.allclose(grad_HS3.reshape((-1,4,4)), grad_jax)
    hvp_correct = jnp.allclose(hvp_HS3.reshape((-1,4,4)), hvp_jax)

    print("\tCost correct: ", cost_correct)
    print("\tGradient correct: ", gradient_correct)
    print("\tHVP correct: ", hvp_correct)
    assert cost_correct
    assert gradient_correct
    assert hvp_correct

    print("All costs are the same: ", cost_HS1, cost_HS2, cost_HS3)


def test_compute_riemannian_loss_gradient_hvp_HST_all_functions():
    print("****** test_compute_riemannian_loss_gradient_hvp_HST_all_functions ******")
    nsites, nbatch, chi_max = 6, 5, 16
    rel_tol, abs_tol = 1e-11, 1e-14

    # ------ Create a random brickwall circuit ------
    nlayers = 3
    G_layers, qubits_in_layer = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Z_layers, _ = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Gs = jnp.asarray(util.flatten(G_layers))
    Zs = jnp.asarray(util.flatten(Z_layers))
    qubits = util.flatten(qubits_in_layer)
    Zs = riemannian_manifold.project_to_tangent_space(Gs, Zs)

    # ------ Create random bra and ket environments ------
    ket0 = mps.get_random_mps_stacked(keys[15], nsites, nbatch)
    bra0 = mps.get_random_mps_stacked(keys[16], nsites, nbatch)
    ket0_padded = tn_helpers.pad_mps_to_max_bonddim(ket0, chi_max)
    bra0_padded = tn_helpers.pad_mps_to_max_bonddim(bra0, chi_max)

    # ------ loss ------
    cost_HS1 = objective_function.compute_loss_HST(
        ket0_padded, bra0_padded, Gs, qubits, True,  chi_max, rel_tol, abs_tol)
            
    # ------ loss, gradient ------
    cost_HS2, grad_HS2 = objective_function.compute_riemannian_loss_gradient_HST(
        ket0_padded, bra0_padded, Gs, qubits, True, chi_max, rel_tol, abs_tol)
            
    # ------ loss, gradient, hvp ------
    cost_HS3, grad_HS3, hvp_HS3 = objective_function.compute_riemannian_loss_gradient_hvp_HST(
        ket0_padded, bra0_padded, Gs, Zs, qubits, True, chi_max, rel_tol, abs_tol)
    
    # ------ hvp ------
    hvp_HS4 = objective_function.compute_riemannian_hvp_HST(
        ket0_padded, bra0_padded, Gs, Zs, qubits, True, chi_max, rel_tol, abs_tol)
    
    gradients_the_same = jnp.allclose(grad_HS2, grad_HS3)
    hvps_the_same = jnp.allclose(hvp_HS3, hvp_HS4)

    print("All costs are the same: ", cost_HS1, cost_HS2, cost_HS3)
    print("All gradients are the same: ", gradients_the_same)
    print("All HVPs are the same: ", hvps_the_same)
    assert gradients_the_same
    assert hvps_the_same

    # ------ Test looping ------
    cost = objective_function.compute_loss_HST_loop(ket0_padded, bra0_padded, Gs, qubits, True, chi_max, rel_tol, abs_tol)
    print("Cost: ", cost)


def test_riemannian_projection():
    print("****** test_riemannian_projection ******")
    nsites, I, nbatch, chi_max = 6, jnp.eye(2), 5, 16

    # ------ Create a random brickwall circuit ------
    nlayers = 3
    G_layers, qubits_in_layer = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Z_layers, _ = brickwall_circuit.get_random_layers(nsites, nlayers, first_layer_odd=True)
    Gs = jnp.asarray(util.flatten(G_layers))
    Zs = jnp.asarray(util.flatten(Z_layers))
    qubits = util.flatten(qubits_in_layer)
    G_gates = brickwall_circuit.get_matrices(Gs)
    Z_gates = brickwall_circuit.get_matrices(Zs)
    Zs = riemannian_manifold.project_to_tangent_space(Gs, Zs)
    Z_gates = brickwall_circuit.get_matrices(Zs)

    # ------ Create random bra and ket environments ------
    ket0 = mps.get_random_mps_stacked(keys[15], nsites, nbatch)
    bra0 = mps.get_random_mps_stacked(keys[16], nsites, nbatch)
    ket0_padded = tn_helpers.pad_mps_to_max_bonddim(ket0, chi_max)
    bra0_padded = tn_helpers.pad_mps_to_max_bonddim(bra0, chi_max)

    # ------ JAX ------
    def f(gates):  # Frobenius
        res = 0
        for _ket0,_bra0 in zip(ket0_padded, bra0_padded):
            ket1 = util.tensor_product(list(gates[:3])) @ mps.get_vector(_ket0)
            ket2 = util.tensor_product([I]+list(reversed(gates[3:5]))+[I]) @ ket1
            ket3 = util.tensor_product(list(gates[5::])) @ ket2
            res += jnp.abs(jnp.dot(mps.get_vector(_bra0), ket3))**2
        return 1 - res/nbatch
    grad_f_HS = jax.grad(f, holomorphic=False)
    riem_grad = lambda g: riemannian_manifold.project_to_tangent_space(
        g.reshape((-1,2,2,2,2)), grad_f_HS(g.reshape((-1,4,4))).conj().reshape((-1,2,2,2,2))).reshape((-1,4,4))

    def hvp_fd(G, xi, grad_R, eps=1e-5):
        # grad at base
        g0 = grad_R(G)
        # grad at perturbed point
        G_eps = riemannian_manifold.retract_to_manifold(G, eps*xi)
        g_eps = grad_R(G_eps)
        # difference quotient, reprojected back to tangent at G
        res = (g_eps - g0) / eps
        return riemannian_manifold.project_to_tangent_space(G.reshape((-1,2,2,2,2)), res.reshape((-1,2,2,2,2)))

    hvp_FD = hvp_fd(G_gates, Z_gates, riem_grad, eps=1e-7).reshape((-1,4,4))
    
    # ------ FoR ------
    cost_HS, grad_HS, hvp_HS = objective_function.compute_riemannian_loss_gradient_hvp_HST(
        ket0_padded, bra0_padded, Gs, Zs, qubits, first_layer_increasing=True, truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14)
           
    hvp_correct = jnp.allclose(util.get_matrices(hvp_HS), hvp_FD)
    
    print("\tHVP correct: ", hvp_correct)
    assert hvp_correct 


def main():
    test_get_gates_per_layer()
    print()
    test_forward_pass_first_order()
    print()
    test_forward_pass_second_order()
    print()
    test_compute_loss_gradient_F()
    print()
    test_compute_loss_HST()
    print()
    test_compute_loss_gradient_HST()
    print()
    test_compute_loss_gradient_hvp_F()
    print()
    test_compute_loss_gradient_hvp_HST()
    print()
    test_riemannian_projection()
    print()
    test_compute_loss_gradient_hvp_HST_all_functions()
    print()
    test_compute_riemannian_loss_gradient_hvp_HST_all_functions()
    


if __name__ == "__main__":
    main()