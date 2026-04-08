import jax.numpy as jnp
from jax import random

from rqcopt_mps import *

key = random.PRNGKey(0)
keys = random.split(key, 10)


def test_get_vector():
    print(5*"*", "Test get_vector()", 5*"*")
    nsites = 6
    max_chi = 8

    # ------ Create single MPS ------
    psi = mps.get_random_mps(keys[0], nsites)
    psi_vec_orig = mps.get_vector(psi)

    # ------ Pad the MPS with zeros ------
    psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi, max_chi)
    psi_padded_vec = mps.get_vector(psi_padded)
   
    print("\tGet vector is correct: ", jnp.allclose(psi_padded_vec, psi_vec_orig))
    assert jnp.allclose(psi_padded_vec, psi_vec_orig)


def test_conj():
    print(5*"*", "Test conj()", 5*"*")
    nsites = 6
    max_chi=8
    psi = mps.get_random_mps(keys[1], nsites)
    psi_vec = mps.get_vector(psi)
    psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi, max_chi)
    psi_conj = psi_padded.conj()

    print("\tConjugate MPS correct: ", jnp.allclose(psi_vec.conj(), mps.get_vector(psi_conj)))
    assert jnp.allclose(psi_vec.conj(), mps.get_vector(psi_conj))


def test_get_haar_random_product_state_stacked():
    print(5*"*", "Test get_haar_random_product_state_stacked()", 5*"*")
    nsites = 6
    nsamples = 5
    psis = mps.get_haar_random_product_state_stacked(keys[2], nsites, nsamples)
    vecs = mps.get_vector_stacked(psis)
   
    print("\tHaar random MPS are different: ", ~jnp.allclose(vecs[0], vecs[1]))
    assert ~jnp.allclose(vecs[0], vecs[1])


def test_increasing_QR_sweep():
    print(5*"*", "Test increasing_QR_sweep", 5*"*")
    # ------ Create MPS ------
    nsites = 6
    max_chi=8
    psi = mps.get_random_mps(keys[3], nsites)
    V1 = mps.get_vector(psi).copy()  # Vector before canoncalization
    psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi, max_chi)
    padded_shape = psi_padded.shape

    # ------ Apply QR sweep ------
    psi_padded = mps.increasing_QR_sweep(psi_padded)  # Canonicalize without truncation
    V2 = mps.get_vector(psi_padded)  # Vector after canonicalization

    print("\tIncreasing QR sweep does not change MPS: ", jnp.allclose(V1,V2))
    print("\tResulting MPS is left canonical: ", mps.is_left_canonical(psi_padded))
    print("\tThe padded result has the correct shape: ", all(jnp.allclose(a,b) for a,b in zip(psi_padded.shape, padded_shape)))
    assert jnp.allclose(V1,V2)
    assert mps.is_left_canonical(psi_padded)
    assert all(jnp.allclose(a,b) for a,b in zip(psi_padded.shape, padded_shape))

    
def test_decreasing_RQ_sweep():
    print(5*"*", "Test decreasing_RQ_sweep", 5*"*")
    # ------ Create MPS ------
    nsites = 6
    max_chi=8
    psi = mps.get_random_mps(keys[4], nsites)
    V1 = mps.get_vector(psi).copy()  # Vector before canoncalization
    psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi, max_chi)
    padded_shape = psi_padded.shape

    # ------ Apply RQ sweep ------
    psi_padded = mps.decreasing_RQ_sweep(psi_padded)  # Canonicalize without truncation
    V2 = mps.get_vector(psi_padded).copy()  # Vector after canonicalization

    print("\tDecreasing RQ sweep does not change MPS: ", jnp.allclose(V1,V2))
    print("\tResulting MPS is right canonical: ", mps.is_right_canonical(psi_padded))
    print("\tThe padded result has the correct shape: ", all(jnp.allclose(a,b) for a,b in zip(psi_padded.shape, padded_shape)))
    assert jnp.allclose(V1,V2)
    assert mps.is_right_canonical(psi_padded)
    assert all(jnp.allclose(a,b) for a,b in zip(psi_padded.shape, padded_shape))


def test_canonicalize_local_tensor():
    print(5*"*", "Test canonicalize_local_tensor", 5*"*")
    # ------ Create MPS ------
    nsites = 6
    chi_max = 8
    psi = mps.get_random_mps(keys[5], nsites)  # Left canonicalize
    psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi, chi_max)
    padded_shape = psi_padded.shape

    site = 3  # Site to be canonicalized
    initial_tensor_is_rc = tn_helpers.local_tensor_is_right_canonical(psi_padded[site])
    psi_padded = mps.right_canonicalize_local_tensor_in_mps(psi_padded, site)
    tensor_is_rc = tn_helpers.local_tensor_is_right_canonical(psi_padded[site])
    represents_the_same_state = jnp.allclose(mps.get_vector(psi), mps.get_vector(psi_padded))
    has_correct_shape = all(jnp.allclose(a,b) for a,b in zip(psi_padded.shape, padded_shape))

    print("\tLocal tensor is right canonical (expect False): ", initial_tensor_is_rc)
    print("\tLocal tensor is right canonical (expect True): ", tensor_is_rc)
    print("\tStill represents the same state: ", represents_the_same_state)
    print("\tThe padded result has the correct shape: ", has_correct_shape)
    assert ~initial_tensor_is_rc
    assert tensor_is_rc
    assert represents_the_same_state
    assert has_correct_shape


    initial_tensor_is_lc = tn_helpers.local_tensor_is_left_canonical(psi_padded[site])
    psi_padded = mps.left_canonicalize_local_tensor_in_mps(psi_padded, site)
    tensor_is_lc = tn_helpers.local_tensor_is_left_canonical(psi_padded[site])
    represents_the_same_state = jnp.allclose(mps.get_vector(psi), mps.get_vector(psi_padded))
    has_correct_shape = all(jnp.allclose(a,b) for a,b in zip(psi_padded.shape, padded_shape))

    print("\n\tLocal tensor is left canonical (expect False): ", initial_tensor_is_lc)
    psi_padded = mps.left_canonicalize_local_tensor_in_mps(psi_padded, site)
    print("\tLocal tensor is left canonical (expect True): ", tensor_is_lc)
    print("\tStill represents the same state: ", represents_the_same_state)
    print("\tThe padded result has the correct shape: ", has_correct_shape)
    assert ~initial_tensor_is_lc
    assert tensor_is_lc
    assert represents_the_same_state
    assert has_correct_shape


def test_is_mixed_canonical():
    print(5*"*", "Test is_mixed_canonical", 5*"*")
    nsites = 6
    chi_max = 8

    print("\n\tUse QR-RQ sweep to get mixed canonical form")
    # ------ Create MPS ------
    psi = mps.get_random_mps(keys[6], nsites)
    psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi, chi_max)
    padded_shape = psi_padded.shape

    # ------ Create mixed canonical form ------
    psi_padded = mps.increasing_QR_sweep(psi_padded, istop=3)
    psi_padded = mps.decreasing_RQ_sweep(psi_padded, istop=3)
    is_mixed, _, _ = mps.is_mixed_canonical(psi_padded, 3)
    has_correct_shape = all(jnp.allclose(a,b) for a,b in zip(psi_padded.shape, padded_shape))

    print("\tExpectation: Canonical center at local tensor 3")
    print("\tIs mixed: ", is_mixed, "\t as expected at local tensor: 3")
    print("\tThe padded result has the correct shape: ", has_correct_shape)
    assert is_mixed
    assert has_correct_shape

    print("\n\tUse RQ-QR sweep to get mixed canonical form")
    # ------ Create MPS ------
    psi = mps.get_random_mps(keys[7], nsites)
    psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi, chi_max)
    padded_shape = psi_padded.shape

    # ------ Create mixed canonical form ------
    psi_padded = mps.decreasing_RQ_sweep(psi_padded)
    psi_padded = mps.increasing_QR_sweep(psi_padded, istop=4)
    is_mixed, _, _ = mps.is_mixed_canonical(psi_padded, 4)
    has_correct_shape = all(jnp.allclose(a,b) for a,b in zip(psi_padded.shape, padded_shape))

    print("\tExpectation: Canonical center at local tensor 4")
    print("\tIs mixed: ", is_mixed, "\tas expected at local tensor: 4")
    print("\tThe padded result has the correct shape: ", has_correct_shape)
    assert is_mixed
    assert has_correct_shape


def test_canonicalize_to_center():
    print(5*"*", "Test canonicalize_to_center", 5*"*")
    # ------ Create MPS ------
    nsites = 6
    chi_max = 8
    psi = mps.get_random_mps(keys[8], nsites)
    psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi, chi_max)
    padded_shape = psi_padded.shape

    # ------ Canonicalize to center ------
    psi_padded = mps.canonicalize_to_center(psi_padded, 3)
    is_mixed, _, _ = mps.is_mixed_canonical(psi_padded, 3)
    has_correct_shape = all(jnp.allclose(a,b) for a,b in zip(psi_padded.shape, padded_shape))

    print("\tMPS is mixed-canonicalized: ", is_mixed)
    print("\tThe padded result has the correct shape: ", has_correct_shape)
    assert is_mixed
    assert has_correct_shape


def test_increasing_SVD_sweep():
    print(5*"*", "Test increasing_SVD_sweep", 5*"*")
    # ------ Create MPS ------
    nsites = 6
    chi_max = 32
    psi = mps.get_random_mps(keys[9], nsites)
    psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi, chi_max)
    padded_shape = psi_padded.shape

    psi_padded = mps.decreasing_RQ_sweep(psi_padded)  # Canonicalize without truncation
    V1 = mps.get_vector(psi_padded).copy()  # Vector before canoncalization

    # ------ Apply SVD sweep ------
    psi_padded = mps.increasing_SVD_sweep(
        psi_padded, max_bondim=chi_max, 
        rel_tol=1e-16, abs_tol=1e-16
        )  # Canonicalize without truncation
    V2 = mps.get_vector(psi_padded).copy()  # Vector after canonicalization

    mps_unchanged = jnp.allclose(V1,V2)
    has_correct_shape = all(jnp.allclose(a,b) for a,b in zip(psi_padded.shape, padded_shape))

    print("\tIncreasing SVD sweep does not change MPS: ", mps_unchanged)
    print("\tThe padded result has the correct shape: ", has_correct_shape)
    assert mps_unchanged
    assert has_correct_shape

    # ------ Test normalization ------
    psi = mps.get_random_mps(keys[3], nsites)
    psi_padded = tn_helpers.pad_mps_to_max_bonddim(psi, chi_max)
    padded_shape = psi_padded.shape

    psi_padded = mps.decreasing_RQ_sweep(psi_padded)  # Canonicalize without truncation
    V1 = mps.get_vector(psi_padded).copy()  # Vector before canoncalization

    psi_padded = mps.increasing_SVD_sweep_normalize(
        psi_padded, max_bondim=chi_max, 
        rel_tol=1e-16, abs_tol=1e-16
        )  # Canonicalize without truncation
    V2 = V1/jnp.linalg.norm(V1)  # Vector after canonicalization

    normalization_correct = jnp.allclose(mps.get_vector(psi_padded),V2)
    shape_correct = all(jnp.allclose(a,b) for a,b in zip(psi_padded.shape, padded_shape))

    print("\tIncreasing SVD sweep normalizes correctly: ", normalization_correct)
    print("\tThe padded result has the correct shape: ", shape_correct)
    assert normalization_correct
    assert shape_correct


def test_add_mps():
    print(5*"*", "Test add_mps", 5*"*")
    nsites = 8
    chi_max = 64

    ket1 = mps.get_random_mps(keys[9], nsites)
    ket2 = mps.get_random_mps(keys[11], nsites)
    res = mps.get_vector(ket1)+mps.get_vector(ket2)

    ket1_padded = tn_helpers.pad_mps_to_max_bonddim(ket1, chi_max)
    ket2_padded = tn_helpers.pad_mps_to_max_bonddim(ket2, chi_max)

    # ------ Addition in decreasing order ------
    test1, _ = mps_addition.add_mps(
        ket1_padded, ket2_padded, canonical_center=int(nsites-1), 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14
        )
    adding_correct = jnp.allclose(res, mps.get_vector(test1))
    print("\tAdding is correct decreasing: ", adding_correct)
    assert adding_correct

    # ------ Addition in increasing order ------
    ket1_padded, ket2_padded = mps.right_canonicalize(ket1_padded), mps.right_canonicalize(ket2_padded)
    test2, _ = mps_addition.add_mps(
        ket1_padded, ket2_padded, canonical_center=0, 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14
        ) 
    adding_correct = jnp.allclose(res, mps.get_vector(test2))
    print("\tAdding is correct increasing: ", adding_correct)
    assert adding_correct

    # ------ Addition in increasing dominant order ------
    ket1_padded = mps.canonicalize_to_center(ket1_padded, 6)
    ket2_padded = mps.canonicalize_to_center(ket2_padded, 6)
    test3, _ = mps_addition.add_mps(
        ket1_padded, ket2_padded, canonical_center=6, 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14
        )
    adding_correct = jnp.allclose(res, mps.get_vector(test3))
    print("\tAdding is correct increasing dominant: ", adding_correct)
    assert adding_correct

    # ------ Addition in decreasing dominant order ------
    ket1_padded = mps.canonicalize_to_center(ket1_padded, 3)
    ket2_padded = mps.canonicalize_to_center(ket2_padded, 3)
    test4, _ = mps_addition.add_mps(
        ket1_padded, ket2_padded, canonical_center=3, 
        truncation_dim=chi_max, rel_tol=1e-11, abs_tol=1e-14
        )
    adding_correct = jnp.allclose(res, mps.get_vector(test4))
    print("\tAdding is correct decreasing dominant: ", adding_correct)
    assert adding_correct
    
  
def test_compute_partial_inner_product_from_mps():
    print(5*"*", "Test compute_partial_inner_product_from_mps", 5*"*")
    def _partial_inner_product(phi, psi, keep, n):
        k = len(keep)
        assert psi.shape == (2**n,) and phi.shape == (2**n,)
        
        all_indices = list(range(n))
        trace = [i for i in all_indices if i not in keep]
        
        # Reshape to n-qubit tensor
        psi_tensor, phi_tensor = psi.reshape([2] * n), phi.reshape([2] * n)
        # Permute axes: [trace..., keep...]
        perm = trace + list(keep)
        psi_perm = jnp.transpose(psi_tensor, perm)
        phi_perm = jnp.transpose(phi_tensor, perm)
        # Collapse to shape: (2**(n-k), 2**k)
        env_dim = 2 ** (n - k)
        kept_dim = 2 ** k
        psi_reshaped = psi_perm.reshape((env_dim, kept_dim))
        phi_reshaped = phi_perm.reshape((env_dim, kept_dim))
        # Contract over environment
        result = jnp.einsum('ai,aj->ij', phi_reshaped, psi_reshaped)
        # Reshape to tensor with 2 indices per qubit kept
        return result.reshape([2] * (2 * k))
    
    # ------ Create MPS ------
    nsites = 6
    chi_max = 32
    ket = mps.get_random_mps(keys[13], nsites)
    bra = mps.get_random_mps(keys[14], nsites)
    ket_padded = tn_helpers.pad_mps_to_max_bonddim(ket, chi_max)
    bra_padded = tn_helpers.pad_mps_to_max_bonddim(bra, chi_max)

    computation_correct = []

    # ------ Compute partial inner products for various cut-out qubits ------
    for cut_out_qubits in [jnp.array([i, i + 1], dtype=jnp.int32) for i in range(nsites-1)]:
        test = mps.compute_partial_inner_product_from_mps(bra_padded, ket_padded, cut_out_qubits)
        test2 = _partial_inner_product(mps.get_vector(bra), mps.get_vector(ket),
                                       keep=cut_out_qubits, n=nsites)
        computation_correct.append(jnp.allclose(test,test2))
    computations_all_correct = all(computation_correct)

    print("\tEverything correct: ", computations_all_correct)
    assert computations_all_correct


def test_compute_inner_product_from_mps():
    print(5*"*", "Test compute_inner_product_from_mps", 5*"*")
    # ------ Create MPS ------
    nsites = 6
    chi_max = 32
    ket = mps.get_random_mps(keys[13], nsites)
    bra = mps.get_random_mps(keys[14], nsites)
    ket_padded = tn_helpers.pad_mps_to_max_bonddim(ket, chi_max)
    bra_padded = tn_helpers.pad_mps_to_max_bonddim(bra, chi_max)

    # ------ Compute inner product ------
    test = mps.compute_inner_product_from_mps(ket_padded, bra_padded)
    test2 = jnp.dot(mps.get_vector(bra), mps.get_vector(ket))
    is_correct = jnp.allclose(test, test2)

    print("\tEverything correct: ", is_correct)
    assert is_correct


def test_compute_fidelity():
    print(5*"*", "Test compute_fidelity", 5*"*", )
    # ------ Create MPS ------
    nsites = 6
    chi_max = 32
    ket = mps.get_random_mps(keys[13], nsites)
    bra = mps.get_random_mps(keys[14], nsites)
    ket_padded = tn_helpers.pad_mps_to_max_bonddim(ket, chi_max)
    bra_padded = tn_helpers.pad_mps_to_max_bonddim(bra, chi_max)

    # ------ Compute fidelity ------
    test = mps.compute_fidelity(ket_padded, bra_padded)
    test2 = jnp.abs(jnp.dot(mps.get_vector(ket), mps.get_vector(bra)))**2
    is_correct = jnp.allclose(test, test2)

    print("\tEverything correct: ", is_correct)
    assert is_correct


def main():
    test_get_vector()
    print()
    test_conj()
    print()
    test_get_haar_random_product_state_stacked()
    print()
    test_increasing_QR_sweep()
    print()
    test_decreasing_RQ_sweep()
    print()
    test_canonicalize_local_tensor()
    print()
    test_canonicalize_to_center()
    print()
    test_is_mixed_canonical()
    print()
    test_increasing_SVD_sweep()
    print()
    test_add_mps()
    print()
    test_compute_partial_inner_product_from_mps()
    print()
    test_compute_inner_product_from_mps()
    print()
    test_compute_fidelity()


if __name__ == "__main__":
    main()