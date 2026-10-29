import jax
import jax.numpy as jnp
from jax import config

from .util import qr, rq

config.update("jax_enable_x64", True)


def compress_SVD(u, s, vh, max_bondim, rel_tol, abs_tol):
    '''
    Discard dimensions larger than max_bondim.
    '''

    u_new = u[..., :max_bondim]
    vh_new = vh[:max_bondim]
    s_new = s[:max_bondim]

    # Apply cutoff
    threshold = jnp.maximum(abs_tol, jnp.max(s) * rel_tol)
    s_clean = jnp.where(s_new > threshold, s_new, 0.0)

    # # Compute lost information
    # total_power = jnp.sum(s**2)
    # kept_power = jnp.sum(s_clean**2)
    # info_loss = (total_power - kept_power) / (total_power + 1e-16)

    return u_new, s_clean, vh_new


def compress_SVD_padded(u, s, v, max_bondim, rel_tol, abs_tol):
    """
    Truncates singular values based on both index (max_bondim) and
    value (hybrid cutoff), while keeping array shapes static.
    """
    D = s.shape[0]
    max_bondim = jnp.clip(max_bondim, 0, D)

    # mask for which singular directions to keep
    idxs = jnp.arange(D)
    index_mask = (idxs < max_bondim).astype(s.dtype)

    # Find the hybrid threshold based on the largest singular value
    threshold = jnp.maximum(abs_tol, jnp.max(s) * rel_tol)
    value_mask = (s > threshold).astype(s.dtype)

    # A direction is only kept if it satisfies BOTH conditions
    combined_mask = index_mask * value_mask

    s_new = s * combined_mask
    u_new = u * combined_mask[None, :]
    v_new = combined_mask[:, None] * v

    # # Compute lost information
    # total_power = jnp.sum(s**2)
    # kept_power = jnp.sum(s_new**2)
    # info_loss = (total_power - kept_power) / (total_power + 1e-16)

    return u_new, s_new, v_new


def local_tensor_is_right_canonical(tensor):
    """
    Check if local tensor is left canonical.
    """
    l, p, r = tensor.shape
    tensor_reshaped = tensor.reshape(l, p * r)
    product = jnp.dot(tensor_reshaped, tensor_reshaped.T.conj())
    deviation = jnp.linalg.norm(product - jnp.eye(l))
    tol = 1e-10
    return deviation < tol


def local_tensor_is_left_canonical(tensor):
    """
    Check if local tensor is right canonical.
    """
    l, p, r = tensor.shape
    tensor_reshaped = tensor.reshape(l * p, r)
    product = jnp.dot(tensor_reshaped.conj().T, tensor_reshaped)
    err = jnp.linalg.norm(product - jnp.eye(r, dtype=product.dtype), ord='fro')
    tol = 1e-10
    return err < tol


# A single padded MPS
mps_tensors_are_left_canonical = jax.vmap(local_tensor_is_left_canonical)
mps_tensors_are_right_canonical = jax.vmap(local_tensor_is_right_canonical)

# Check, whether a chosen site is canonicalized for a batch of MPSs


def left_canonicalize_local_tensor(A):
    # Correct shapes
    tensor_shape = A.shape
    chi_max, p, _ = tensor_shape
    matrix_shape = (chi_max*p, chi_max)

    # Reshape local tensor to matrix
    C = A.reshape(matrix_shape)
    # Perform QR
    Q, R = qr(C)
    # Restore MPS form
    Q = Q.reshape(tensor_shape)

    return Q, R


def right_canonicalize_local_tensor(A):
    ''' A is the local tensor to be split into RQ. '''

    # Correct shapes
    chiL, p, chiR = A.shape

    # Reshape local tensor to matrix
    C = A.reshape((chiL, p*chiR))
    # Perform RQ
    R, Q = rq(C)

    # Restore MPS form
    Q = Q.reshape((chiL, p, chiR))

    return Q, R


def pad_mps_to_max_bonddim(As, chi_max, keep_axes=False):
    """
    Pad a list of (batched) MPS tensors to a common virtual bond dimension chi_max.

    """

    # Whether MPS is batched
    consider_batch = bool(len(As[0].shape) > 3)

    padded_mps = []
    for A in As:
        local_shape = A.shape
        Dl, d, Dr = local_shape[-3], local_shape[-2], local_shape[-1]
        if consider_batch:
            A_padded = jnp.zeros((local_shape[0], chi_max, d, chi_max), dtype=A.dtype)
            A_padded = A_padded.at[:, :Dl, :, :Dr].set(A)
        else:
            A_padded = jnp.zeros((chi_max, d, chi_max), dtype=A.dtype)
            A_padded = A_padded.at[:Dl, :, :Dr].set(A)

        padded_mps.append(A_padded)

    if consider_batch and not keep_axes:
        padded_mps = jnp.asarray(padded_mps).transpose(1, 0, 2, 3, 4)

    return jnp.asarray(padded_mps)


def unpad_mps_to_list(As):
    """
    As: (nsites, chi, 2, chi) padded MPS (open boundary).
    Returns: list of length nsites with tensors of shape (dl_i, 2, dr_i).
    Not jittable (variable shapes).
    """
    nsites, chi, d, chi2 = As.shape
    assert d == 2 and chi2 == chi

    # chi_k^max for k=0..nsites
    ks = jnp.arange(nsites + 1, dtype=jnp.int64)
    exp = jnp.minimum(ks, nsites - ks)
    chi_phys = jnp.minimum(jnp.int64(chi), (jnp.int64(1) << exp))  # (nsites+1,)

    dl = chi_phys[:-1]
    dr = chi_phys[1:]

    # Build variable-shaped list (Python loop)
    out = []
    for i in range(nsites):
        dli = int(dl[i])
        dri = int(dr[i])
        out.append(As[i, :dli, :, :dri])  # (dli, 2, dri)
    return out
