import jax.numpy as jnp
from jax import random

from rqcopt_mps import *

key = random.PRNGKey(0)
keys = random.split(key, 20)


def test_merge_gate_with_mps():
    print("****** Test merge_gate_with_mps ******")
    nsites = 6
    I = jnp.eye(2)
    chi_max = 32

    print("\tMerge gate with MPS in an increasing sweep")
    computation_correct = []
    for i in range(0,nsites-1):
        qubits = jnp.asarray([i,i+1], dtype=jnp.int32)
        G = brickwall_circuit.get_random_TwoQubitGate()
        psi = mps.get_random_mps(keys[0], nsites)
        psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi, chi_max)
        psi_padded = mps.canonicalize_to_center(psi_padded, i)
        phi_TN, _ = tn_methods.merge_gate_with_mps_wrapper(
            psi_padded, G, qubits, left_canonical=True, 
            truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14
            )
        center = i+2
        if i<nsites-2: print("\t\tCorrectly canoicalized", mps.is_mixed_canonical(phi_TN, center)[0])
        phi_mat = util.tensor_product(
            [I for _ in range(qubits[0])] 
            + [G.reshape((4,4))] 
            + [I for _ in range(qubits[-1]+1, nsites)]
            ) @ mps.get_vector(psi)
        computation_correct.append(jnp.allclose(mps.get_vector(phi_TN), phi_mat))

    print("\t\tCorrectly canoicalized", mps.is_mixed_canonical(phi_TN, nsites-3)[0])
        
    print("\t Merged correct: ", all(computation_correct))
    assert all(computation_correct)

    print("\tMerge gate with MPS in a decreasing sweep")
    computation_correct = []
    for i in reversed(range(0,nsites-1)):
        qubits = jnp.asarray([i,i+1], dtype=jnp.int32)
        G = brickwall_circuit.get_random_TwoQubitGate()
        psi = mps.get_random_mps(keys[0], nsites)
        psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi, chi_max)
        psi_padded = mps.canonicalize_to_center(psi_padded, i)
        phi_TN, _ = tn_methods.merge_gate_with_mps_wrapper(
            psi_padded, G, qubits, left_canonical=False, 
            truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14
            )
        center = i-1
        if i>0: print("\t\tCorrectly canoicalized", mps.is_mixed_canonical(phi_TN, center)[0])
        phi_mat = util.tensor_product(
            [I for _ in range(qubits[0])] 
            + [G.reshape((4,4))] 
            + [I for _ in range(qubits[-1]+1, nsites)]
            ) @ mps.get_vector(psi)
        computation_correct.append(jnp.allclose(mps.get_vector(phi_TN), phi_mat))

    print("\t\tCorrectly canoicalized", mps.is_mixed_canonical(phi_TN, 2)[0])
        
    print("\tMerged correct: ", all(computation_correct))
    assert all(computation_correct)


def test_merge_layer_with_mps():
    print("****** Test merge_layer_with_mps ******")
    nsites = 6
    I = jnp.eye(2)
    chi_max = 32

    psi = mps.get_random_mps(keys[1], nsites)
    psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi, chi_max)

    # ------- Odd layer -------
    qubits = jnp.asarray([[0,1], [2,3], [4,5]], dtype=jnp.int32)
    layer = [brickwall_circuit.get_random_TwoQubitGate() for q in qubits]
    phi_mat = util.tensor_product([gate.reshape((4,4)) for gate in layer]) @ mps.get_vector(psi)
    
    # _______ Increasing sweep _______
    psi_padded = mps.decreasing_RQ_sweep(psi_padded)
    phi_TN = tn_methods.merge_layer_with_mps(
        psi_padded, layer, qubits, increasing_merging_order=True, 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14)
    
    sweep_correct = jnp.allclose(mps.get_vector(phi_TN), phi_mat)
    mixed_canonical = mps.is_mixed_canonical(phi_TN, qubits[-1][0]-1)[0]
    print("\tOdd layer, increasing sweep correct: ", sweep_correct, 
          '\tMixed-canonical: ', mixed_canonical)
    assert sweep_correct
    assert mixed_canonical

    # _______ Decreasing sweep _______
    psi_padded = mps.increasing_QR_sweep(psi_padded)
    phi_TN = tn_methods.merge_layer_with_mps(
        psi_padded, layer, qubits, increasing_merging_order=False,
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14)
    
    sweep_correct = jnp.allclose(mps.get_vector(phi_TN), phi_mat)
    mixed_canonical = mps.is_mixed_canonical(phi_TN, qubits[0][-1]+1)[0]
    print("\tOdd layer, decreasing sweep correct: ", sweep_correct, 
          '\tMixed-canonical: ', mixed_canonical)
    assert sweep_correct
    assert mixed_canonical

    # ------- Even layer -------
    qubits = jnp.asarray([[1,2], [3,4]], dtype=jnp.int32)
    layer = [brickwall_circuit.get_random_TwoQubitGate() for q in qubits]
    phi_mat = util.tensor_product([I] + [gate.reshape((4,4)) for gate in layer] + [I]) @ mps.get_vector(psi)

    # _______ Increasing sweep _______
    psi_padded = mps.decreasing_RQ_sweep(psi_padded)
    phi_TN = tn_methods.merge_layer_with_mps(
        psi_padded, layer, qubits, increasing_merging_order=True, 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14)
    
    sweep_correct = jnp.allclose(mps.get_vector(phi_TN), phi_mat)
    left_canonical = mps.is_left_canonical(phi_TN)
    print("\tEven layer, increasing sweep correct: ", sweep_correct, 
          '\tLeft-canonical: ', left_canonical)
    assert sweep_correct
    assert left_canonical

    # _______ Decreasing sweep _______
    psi_padded = mps.increasing_QR_sweep(psi_padded)
    phi_TN = tn_methods.merge_layer_with_mps(
        psi_padded, layer, qubits, increasing_merging_order=False, 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14)
    
    sweep_correct = jnp.allclose(mps.get_vector(phi_TN), phi_mat)
    right_canonical = mps.is_right_canonical(phi_TN)
    print("\tEven layer, decreasing sweep correct: ", sweep_correct, 
          '\tRight-canonical: ', right_canonical)
    assert sweep_correct
    assert right_canonical
    

def test_compute_all_intermediate_states():
    print("****** Test compute_all_intermediate_states ******")
    nqubits, I, I2, chi_max = 6, jnp.eye(2), jnp.eye(4), 8

    # ------- Create a random brickwall circuit -------
    nlayers = 2
    first_layer_odd = True
    random_layers, qubits = brickwall_circuit.get_random_layers(nqubits, nlayers, first_layer_odd)

    # ------- Create a random ket -------
    psi_0 = mps.get_random_mps(keys[4], nqubits)
    psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi_0, chi_max)
    intermediate_states = tn_methods.compute_all_intermediate_states(
        random_layers, qubits, psi_padded, increasing_merging_order=True, 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14
        )

    # ------- Compute the explicit result -------
    psi_0 = mps.get_vector(psi_0)
    psi_1 = util.tensor_product([random_layers[0][0].reshape((4,4)),I2,I2]) @ psi_0
    psi_2 = util.tensor_product([I2,random_layers[0][1].reshape((4,4)),I2]) @ psi_1
    psi_3 = util.tensor_product([I2,I2,random_layers[0][-1].reshape((4,4))]) @ psi_2 
    psi_4 = util.tensor_product([I,I2]+[random_layers[-1][0].reshape((4,4))] + [I]) @ psi_3 
    psi_5 = util.tensor_product([I]+[random_layers[-1][-1].reshape((4,4))]+[I2,I]) @ psi_4
    res = [psi_5, psi_4, psi_3, psi_2, psi_1, psi_0][::-1]

    is_correct = all([jnp.allclose(psi, mps.get_vector(phi)) for psi, phi in zip(res, intermediate_states)])
    print("\tLocal tensors are correct for ket: ", is_correct)
    assert is_correct
    


def main():
    test_merge_gate_with_mps()
    print()
    test_merge_layer_with_mps()
    print()
    test_compute_all_intermediate_states()



if __name__ == "__main__":
    main()
