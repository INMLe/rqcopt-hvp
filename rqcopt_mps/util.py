from functools import reduce

import numpy as np
from scipy.optimize import curve_fit
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

def mask_to_zeros(As, threshold=1e-16):
    real_part, imag_part = jnp.real(As), jnp.imag(As)
    threshold = 1e-16
    As = (real_part * (jnp.abs(real_part) >= threshold) +
        1j * imag_part * (jnp.abs(imag_part) >= threshold))
    return As

def tensor_product(operators):
    return reduce(jnp.kron, operators)

def get_trotter_scaling(ts, cost):
    def _func(x, a, b):
        return a + b*x
    x = np.log(np.asarray(ts))
    y = np.log(np.asarray(cost))
    popt, _ = curve_fit(_func, x, y)
    func = lambda x: jnp.exp(popt[0]) *  x**popt[1]
    return jnp.exp(x), func, popt

def flatten(list_of_list):
    return jnp.asarray([element for L in list_of_list for element in L])

def tup(arr):
    return tuple(arr.tolist())

def svd(A):
    return jnp.linalg.svd(A, full_matrices=False)

def qr(A):
    return jnp.linalg.qr(A, 'reduced')

def rq(A):
    """
    Computes the RQ decomposition A = R @ Q for a matrix A (m x n, m <= n)
    using the correct algebraic relationship with JAX's QR and matrix flips.
    """
    # A' = J_n @ A^T @ J_m  (transpose + flip both axes)
    A_prime = jnp.flip(A.swapaxes(-1, -2), axis=(-2, -1))

    # QR decomposition of A'
    Q_QR, R_QR = jnp.linalg.qr(A_prime)

    # R = J_m @ R_QR^T @ J_m  (transpose + flip both axes)
    R = jnp.flip(R_QR.swapaxes(-1, -2), axis=(-2, -1))

    # Q = J_n @ Q_QR^T @ J_m  (transpose + flip both axes)
    Q = jnp.flip(Q_QR.swapaxes(-1, -2), axis=(-2, -1))

    return R, Q

@jax.jit
def transpose_tensors(Gs):
    ''' Transpose an array of gates Gs: (a, i, j, k, l) -> (a, k, l, i, j) '''
    return jnp.transpose(Gs, (0, 3, 4, 1, 2))

def adjoint_tensors(Gs):
    ''' Take the adjoint of an array of gates Gs: (a, i, j, k, l) -> (a, k, l, i, j) '''
    return jnp.transpose(Gs, (0, 3, 4, 1, 2)).conj()

def transpose_matrices(Gs):
    ''' Take the transpose of an array of gates Gs: (a, i, j) -> (a, j, i) '''
    return jnp.transpose(Gs, (0, 2, 1))

@jax.jit
def adjoint_matrices(Gs):
    ''' Take the adjoint of an array of gates Gs: (a, i, j) -> (a, j, i) '''
    return jnp.transpose(Gs, (0, 2, 1)).conj()

@jax.jit
def get_matrices(gates):
    return gates.reshape((-1,4,4))

@jax.jit
def get_tensors(gates):
    return gates.reshape((-1,2,2,2,2))

@jax.jit
def batch_multiplication(As, Bs, Cs):
    return jnp.einsum('eabmn, emnpq, epqcd -> eabcd', As, Bs, Cs)