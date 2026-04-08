import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from .util import qr, rq, svd


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

    return u_new, s_new, v_new

def local_tensor_is_right_canonical(tensor):
    """
    Check if local tensor is left canonical.
    """
    l, p, r = tensor.shape
    tensor_reshaped = tensor.reshape(l, p * r)
    product = jnp.dot(tensor_reshaped, tensor_reshaped.T.conj())
    deviation = jnp.linalg.norm(product - jnp.eye(l))
    tol=1e-10
    return deviation < tol

def local_tensor_is_left_canonical(tensor):
    """
    Check if local tensor is right canonical.
    """
    l, p, r = tensor.shape
    tensor_reshaped = tensor.reshape(l * p, r)
    product = jnp.dot(tensor_reshaped.conj().T, tensor_reshaped)
    err = jnp.linalg.norm(product - jnp.eye(r, dtype=product.dtype), ord='fro')
    tol=1e-10
    return err < tol

# A single padded MPS
mps_tensors_are_left_canonical = jax.vmap(local_tensor_is_left_canonical)
mps_tensors_are_right_canonical = jax.vmap(local_tensor_is_right_canonical)

# Check, whether a chosen site is canonicalized for a batch of MPSs
batch_local_tensor_is_right_canonical = jax.vmap(local_tensor_is_right_canonical,in_axes=0)
batch_local_tensor_is_left_canonical = jax.vmap(local_tensor_is_left_canonical,in_axes=0)

def build_block_diagonal_tensor(A0, A1):
    Dl0, d, Dr0 = A0.shape
    Dl1, _, Dr1 = A1.shape

    # Block diagonal
    top = jnp.concatenate([A0, jnp.zeros((Dl0, d, Dr1))], axis=2)
    bottom = jnp.concatenate([jnp.zeros((Dl1, d, Dr0)), A1], axis=2)
    A_block = jnp.concatenate([top, bottom], axis=0)   # (Dl0+Dl1, d, Dr0+Dr1)
    return A_block

def truncate_and_left_canonicalize_local_tensor(A_block, truncation_dim, rel_tol, abs_tol):
    Dl, d, Dr = A_block.shape
    mat = A_block.reshape(Dl*d, Dr)
    u, s, vh = svd(mat)
    u, s, vh = compress_SVD(u, s, vh, truncation_dim, rel_tol, abs_tol)
    res = u.reshape(Dl, d, u.shape[-1])
    carry = s[:, None] * vh   # contract into next site
    return res, carry

def truncate_and_right_canonicalize_local_tensor(A_block, truncation_dim, rel_tol, abs_tol):
    Dl, d, Dr = A_block.shape
    mat = A_block.reshape(Dl, d*Dr)
    u, s, vh = svd(mat)
    u, s, vh = compress_SVD(u, s, vh, truncation_dim, rel_tol, abs_tol)
    carry = u * s[None, :]    # propagate left
    res = vh.reshape(-1, d, Dr)
    return res, carry

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

batched_left_canonicalize_local_tensor = jax.vmap(
    left_canonicalize_local_tensor, 
    in_axes=(0, None), # 0 for the list of tensors (As), None for the scalar max_bondim
    out_axes=0         # 0 for the output list of tensors (As)
)

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

def pad_local_tensor_to_max_bonddim(A, chi_max):
    """
    Pad a local tensor A of shape (Dl, d, Dr) up to (chi_max, d, chi_max)
    by adding zeros on the right of the bond dimensions.

    This is jittable and does not use Python if-statements.
    If A already has shape (chi_max, d, chi_max), it is returned unchanged.
    """
    Dl, d, Dr = A.shape

    pad_left  = chi_max - Dl
    pad_right = chi_max - Dr

    # pads: ( (before_L, after_L), (before_d, after_d), (before_R, after_R) )
    pads = ((0, pad_left), (0, 0), (0, pad_right))

    A_padded = jnp.pad(A, pads, mode="constant", constant_values=0)
    return A_padded

def get_boundary_vector(chi_max):
    alpha = jnp.zeros(chi_max)
    alpha = alpha.at[0].set(1.0) #jnp.eye(1) 
    return alpha

def pad_mps_to_max_bonddim(As, chi_max, keep_axes=False):
    """
    Pad a list of (batched) MPS tensors to a common virtual bond dimension chi_max.

    """

    # Whether MPS is batched
    consider_batch = bool(len(As[0].shape)>3)
    
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
        padded_mps = jnp.asarray(padded_mps).transpose(1,0,2,3,4)

    return jnp.asarray(padded_mps)

def zero_out_unphysical_bonds(As):
    """
    As: (nsites, chi, 2, chi) padded MPS.
    Zeros entries outside physically allowed dl/dr per site (open boundaries).
    """
    nsites, chi, d, chi2 = As.shape
    assert d == 2 and chi2 == chi

    # Compute chi_k^max = min(2^k, 2^(nsites-k)), then clip to padding chi
    k = jnp.arange(nsites + 1, dtype=jnp.int64)
    exp = jnp.minimum(k, nsites - k)                 # exponent for 2^min(k, n-k)
    chi_phys = jnp.minimum(jnp.int64(chi), (jnp.int64(1) << exp))  # (nsites+1,)

    dl = chi_phys[:-1].astype(jnp.int64)  # (nsites,) left bond limit at each site
    dr = chi_phys[1:].astype(jnp.int64)   # (nsites,) right bond limit at each site

    l = jnp.arange(chi, dtype=jnp.int64)  # (chi,)
    r = jnp.arange(chi, dtype=jnp.int64)  # (chi,)

    mask_l = l[None, :] < dl[:, None]     # (nsites, chi)
    mask_r = r[None, :] < dr[:, None]     # (nsites, chi)

    mask = (mask_l[:, :, None] & mask_r[:, None, :])  # (nsites, chi, chi)
    mask = mask[:, :, None, :]                        # (nsites, chi, 1, chi) -> broadcast over phys dim

    return As * mask.astype(As.dtype)

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