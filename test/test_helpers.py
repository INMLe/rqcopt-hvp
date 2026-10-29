import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.numpy import asarray, zeros_like
from scipy.stats import unitary_group

from rqcopt_mps.tn_helpers import (
    mps_tensors_are_left_canonical, mps_tensors_are_right_canonical,
)
from rqcopt_mps.util import qr

np.random.seed(1234)


def eval_numerical_gradient_complex(f, x, h=1e-5):
    """
    Numerically approximate the gradient of a real-valued function f(x),
    where x is complex-valued. Uses the Wirtinger convention.

    Parameters:
        f : function mapping complex array to real scalar
        x : complex array (jax.numpy)
        h : finite difference step size (float)

    Returns:
        grad : complex-valued gradient (Wirtinger)
    """
    x = asarray(x)
    flat_x = x.ravel()
    grad = []

    for i in range(flat_x.size):
        ei = zeros_like(flat_x).at[i].set(1.0)

        fpos = f((flat_x + h * ei).reshape(x.shape))
        fneg = f((flat_x - h * ei).reshape(x.shape))
        f_up = f((flat_x + 1j * h * ei).reshape(x.shape))
        f_dn = f((flat_x - 1j * h * ei).reshape(x.shape))

        df = ((fpos - fneg) - 1j * (f_up - f_dn)) / (4 * h)
        grad.append(df)

    return 2*asarray(grad).reshape(x.shape).conj()


def get_random_TwoQubitGate():
    return unitary_group.rvs(4).reshape((2, 2, 2, 2))


def get_random_layers(n_sites, n_layers, first_layer_odd, is_TI=False):
    """
    Generate n_layers layers with two-qubit gates expressed
    as tensors --> alternate between odd and even layers.
    """
    layers, qubits_in_layer = [], []
    odd = first_layer_odd
    N_odd_gates, N_even_gates = n_sites//2, n_sites//2  # Number of gates per layer
    if n_sites%2 == 0: N_even_gates -= 1

    for _ in range(n_layers):
        if odd:
            n_gates = N_odd_gates
            iqubit = 0
        else:
            n_gates = N_even_gates
            iqubit = 1
        if is_TI:
            layers += [[get_random_TwoQubitGate()]]
        else:
            layers += [[get_random_TwoQubitGate() for _ in range(n_gates)]]
        _qubits_in_layer = [[iqubit+i, iqubit+i+1] for i in range(0, 2*n_gates, 2)]
        if not odd: _qubits_in_layer = _qubits_in_layer[::-1]
        qubits_in_layer += [_qubits_in_layer]

        odd = not odd

    return layers, qubits_in_layer


def get_tangent_vector(As, Bs):
    """
    Full vector of the tangent state of an augmented MPS (As, Bs):
    the sum over sites i of the MPS with As[i] replaced by Bs[i].
    """
    return sum(get_vector(As.at[i].set(Bs[i])) for i in range(As.shape[0]))


# ----- MPS tools -----

def get_vector(As):
    """
    As: list of potentially padded tensors, each (chi_max, p, chi_max)
    Returns: vector of shape (p**N,)
    """
    # Open boundary conditions
    alpha = jnp.zeros(As[0].shape[0])
    alpha = alpha.at[0].set(1.0)

    # Contract left boundary with first site
    tensor = jnp.einsum('i,iaj->aj', alpha, As[0])  # (p, chi_max)

    for next_tensor in As[1:]:
        tensor = jnp.einsum('aj,jbk->abk', tensor, next_tensor)
        tensor = tensor.reshape(tensor.shape[0] * tensor.shape[1],
                                tensor.shape[2])

    # Contract right boundary
    vector = jnp.einsum('ak,k->a', tensor, alpha)  # (p**N,)

    return vector


def get_vector_stacked(As_stacked):
    return jax.vmap(get_vector)(As_stacked)


def get_mps_from_vector(vec):
    '''
    Decompose a vector into an MPS by means of QR decomposition.
    This will result in a left-canonical form.
    '''

    As = []  # List of local tensors

    N = int(jnp.round(jnp.log2(len(vec))))  # Number of sites
    shape = (2,)*N
    A = vec.reshape(shape)
    A = A[jnp.newaxis, ..., jnp.newaxis]  # Extend by dummy bonds

    for site in range(1, N):
        # Reshape vector to 2D matrix
        shape_1 = A.shape
        shape_2 = (int(jnp.prod(jnp.asarray(shape_1[:2]))),
                   int(jnp.prod(jnp.asarray(shape_1[2:]))))
        B = A.reshape(shape_2)

        # Perform QR
        Q, R = qr(B)

        # Reshape Q and R tot appropriate local tensors
        shape_3 = shape_1[:2] + (Q.shape[-1],)
        E = Q.reshape(shape_3)
        As.append(E)
        shape_4 = (Q.shape[-1],) + shape_1[2:]
        F = R.reshape(shape_4)

        # Append last local tensor
        if site == N-1: As.append(F)
        A = F

    return As


def get_mps_from_vector_stacked(vectors):
    """
    Given a list/array of vectors, return stacked MPS tensors.

    Output format:
        stacked_mps[site] has shape (num_vecs, ...) and contains
        the MPS tensor at this site for each vector.
    """
    # Compute one MPS per vector
    mps_list = [get_mps_from_vector(v) for v in vectors]

    # Group tensors by site
    per_site = list(zip(*mps_list))   # now per_site[i] = (A1_i, A2_i, ..., Ak_i)

    # Stack across vectors
    stacked = [jnp.stack(site_tensors, axis=0) for site_tensors in per_site]

    return stacked


def get_random_mps(key, nsites):
    key_real, key_imag = random.split(key)
    real_part = random.uniform(key_real, shape=(2**nsites,))
    imag_part = random.uniform(key_imag, shape=(2**nsites,))
    vec = real_part + 1j * imag_part

    psi = get_mps_from_vector(vec)
    return psi


def get_random_mps_stacked(key, nsites, nbatch):
    """
    Generate `nbatch` random state vectors and convert them (in batch)
    to stacked MPS tensors returned as a list of per-site arrays.

    Returns:
        As: list of length nsites where As[s].shape == (nbatch, left_bond, 2, right_bond)
    """
    dim = 2 ** nsites
    key_real, key_imag = random.split(key)
    # draw real and imag parts in batch
    real_part = random.normal(key_real, shape=(nbatch, dim))
    imag_part = random.normal(key_imag, shape=(nbatch, dim))
    psis = real_part + 1j * imag_part

    # normalize per state
    norms = jnp.linalg.norm(psis, axis=1, keepdims=True)
    psis = psis / norms

    # convert batch of state vectors -> list of stacked local tensors
    As = get_mps_from_vector_stacked(psis)
    return As


def is_left_canonical(As):
    return jnp.all(mps_tensors_are_left_canonical(As[:-1]))


def is_right_canonical(As):
    return jnp.all(mps_tensors_are_right_canonical(As[1:]))


def is_mixed_canonical(As, center):
    """
    Verify whether `center` is the canonical center of the MPS.
    """

    left_flags = mps_tensors_are_left_canonical(As[:center])
    right_flags = mps_tensors_are_right_canonical(As[center+1:])

    # Conditions for mixed canonical:
    left_ok = jnp.all(left_flags)
    right_ok = jnp.all(right_flags)

    is_correct = left_ok & right_ok
    return is_correct, left_flags, right_flags
