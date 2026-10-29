import jax
import jax.numpy as jnp
from jax import config, lax, random

from .tn_helpers import (
    compress_SVD,
    right_canonicalize_local_tensor,
    compress_SVD_padded,
    left_canonicalize_local_tensor,
    mps_tensors_are_left_canonical,
    mps_tensors_are_right_canonical,
)
from .util import svd, rq

config.update("jax_enable_x64", True)


# -------------------------- Helper functions for MPS --------------------------


def get_haar_random_product_state_stacked(key, nsites, nsamples, d=2):
    """
    Returns: list length nsites, each element has shape (nsamples, 1, d, 1).
    """
    k_re, k_im = random.split(key, 2)

    re = random.normal(k_re, (nsamples, nsites, d))
    im = random.normal(k_im, (nsamples, nsites, d))
    psi = re + 1j * im  # (nsamples, nsites, d)

    psi = psi / jnp.linalg.norm(psi, axis=-1, keepdims=True)

    psi = psi.reshape(nsamples, nsites, 1, d, 1)

    # return list of length nsites, each (nsamples, 1, d, 1)
    return jnp.asarray([psi[:, i, :, :, :] for i in range(nsites)])


# -------------------------- Local tensors in an MPS --------------------------


def split_tensor_via_SVD(B, truncation_dim, rel_tol, abs_tol):

    shape = B.shape
    shape_mat = (shape[0]*shape[1], shape[2]*shape[3])
    B_mat = B.reshape(shape_mat)

    u, s, vh = svd(B_mat)
    u, s, vh = compress_SVD(u, s, vh, truncation_dim, rel_tol, abs_tol)

    return u, s, vh


def split_tensor_into_local_MPS_tensors_left_canonical(B, truncation_dim, rel_tol, abs_tol):
    '''
    Split a tensor of shape (vl,p,p,vr) into two local MPS tensors.
    If left_canonical merge the remainder into the next right local tensor.

    '''

    u, s, vh = split_tensor_via_SVD(B, truncation_dim, rel_tol, abs_tol)
    T2 = s[:, None] * vh

    vl, p1, p2, vr = B.shape
    chi = u.shape[-1]
    A1 = u.reshape((vl, p1, chi))
    A2 = T2.reshape((chi, p2, vr))

    return A1, A2


def split_tensor_into_local_MPS_tensors_right_canonical(B, truncation_dim, rel_tol, abs_tol):
    '''
    Split a tensor of shape (vl,p,p,vr) into two local MPS tensors.
    If left_canonical merge the remainder into the next right local tensor.

    '''

    u, s, vh = split_tensor_via_SVD(B, truncation_dim, rel_tol, abs_tol)
    T1 = u * s[None, :]

    vl, p1, p2, vr = B.shape
    chi = vh.shape[0]
    A1 = T1.reshape(((vl, p1, chi)))
    A2 = vh.reshape((chi, p2, vr))

    return A1, A2


def left_canonicalize_local_tensor_in_mps(As, site):
    ''' A is the local tensor to be split into RQ. '''

    Q, R = left_canonicalize_local_tensor(As[site])
    # Restore MPS form
    As = As.at[site].set(Q)
    A_next = jnp.einsum('ij,jkl->ikl', R, As[site+1])
    As = As.at[site+1].set(A_next)
    return As


def right_canonicalize_local_tensor_in_mps(As, site):
    ''' A is the local tensor to be split into RQ. '''

    Q, R = right_canonicalize_local_tensor(As[site])
    # Restore MPS form
    As = As.at[site].set(Q)
    A_prev = jnp.einsum('ijk,kl->ijl', As[site-1], R)
    As = As.at[site-1].set(A_prev)
    return As


# -------------------------- SWEEPS --------------------------

def _increasing_SVD_sweep_core(As, max_bondim, rel_tol, abs_tol):
    """
    As: (nsites, chi_max, p, chi_max)
    cutoff: Threshold below which singular values are treated as EXACT zeros.
    """
    nsites, chi_max, p, _ = As.shape
    tensor_shape = (chi_max, p, chi_max)
    matrix_shape = (chi_max * p, chi_max)

    max_bondim = jnp.asarray(max_bondim, jnp.int32)
    s0 = jnp.zeros((chi_max,))

    def body_fun(i, carry):
        As, last_s = carry

        # Bring local tensor into matrix form
        A = As[i]
        C = A.reshape(matrix_shape)

        # SVD
        u, s, v = svd(C)

        # Truncation
        u, s, v = compress_SVD_padded(u, s, v, max_bondim, rel_tol, abs_tol)

        # Restore MPS form
        SV = s[:, None] * v

        A_i_new = u.reshape(tensor_shape)

        # Contract cleaned SV into the next site.
        A_ip1_new = jnp.einsum("ij,jak->iak", SV, As[i + 1])

        As = As.at[i].set(A_i_new)
        As = As.at[i + 1].set(A_ip1_new)

        return As, s  # Pass the clean spectrum

    As, s_last = lax.fori_loop(0, nsites - 1, body_fun, (As, s0))
    return As, s_last


def increasing_SVD_sweep(As, max_bondim, rel_tol, abs_tol):
    return _increasing_SVD_sweep_core(As, max_bondim, rel_tol, abs_tol)[0]


def increasing_SVD_sweep_normalize(As, max_bondim, rel_tol, abs_tol):
    ''' Assume that the MPS is in right-canonical form! '''

    As, s = _increasing_SVD_sweep_core(As, max_bondim, rel_tol, abs_tol)

    # Normalize
    nrm = jnp.sqrt(jnp.sum(s**2))
    As = As.at[-1].set(As[-1]/nrm)
    nrm = jnp.abs(nrm)

    return As


def batched_increasing_SVD_sweep_normalize(As, max_bondim, rel_tol, abs_tol):
    return jax.vmap(
    increasing_SVD_sweep_normalize,
    in_axes=(0, None, None, None),
    out_axes=0
)(As, max_bondim, rel_tol, abs_tol)


def increasing_QR_sweep(As, istart=0, istop=1000):
    """
    Left-to-right QR sweep for a padded MPS.
    All tensors have shape (chi_max, p, chi_max).
    QR is applied to the reshaped (chi_max*p, chi_max) matrix.
    The output is re-padded back to (chi_max, p, chi_max).

    As: array with shape (nsites, chi_max, p, chi_max)
    istart, istop: can be Python ints or JAX scalar ints.
    """

    nsites, chi_max, p, _ = As.shape

    # clamp to valid range; convert to JAX ints
    istart = jnp.asarray(istart, jnp.int32)
    istop = jnp.asarray(istop,  jnp.int32)
    istart = jnp.clip(istart, 0, nsites - 1)
    istop = jnp.clip(istop,  0, nsites - 1)

    tensor_shape = (chi_max, p, chi_max)
    matrix_shape = (chi_max * p, chi_max)

    def body_fun(i, As):
        A = As[i]
        C = A.reshape(matrix_shape)

        Q, R = jnp.linalg.qr(C, mode="reduced")
        A_new = Q.reshape(tensor_shape)

        A_next_new = jnp.einsum("ij,jbk->ibk", R, As[i + 1])

        As = As.at[i].set(A_new)
        As = As.at[i + 1].set(A_next_new)
        return As

    # Loop only over the sites that actually need processing
    As = lax.fori_loop(istart, istop, body_fun, As)
    return As


def decreasing_RQ_sweep(As, istart=1000, istop=0):
    """
    Right-to-left RQ sweep for a padded MPS.
    All tensors have shape (chi_max, p, chi_max).
    RQ is applied to the reshaped (chi_max, p*chi_max) matrix.
    The output is re-padded back to (chi_max, p, chi_max).

    As: array with shape (nsites, chi_max, p, chi_max)
    istart: exclusive upper bound (sweep processes sites < istart)
    istop:  inclusive lower bound (sweep processes sites > istop, and site > 0)
    """

    nsites, chi_max, p, _ = As.shape

    # istart in [0, nsites]; istop in [0, nsites-1]
    istart = jnp.asarray(istart, jnp.int32)
    istop = jnp.asarray(istop,  jnp.int32)
    istart = jnp.clip(istart, 0, nsites)
    istop = jnp.clip(istop,  0, nsites - 1)

    # Effective range: sites from istart-1 down to max(istop+1, 1)
    # We also need i > 0 because we access As[i-1].
    lo = jnp.maximum(istop + 1, 1)  # inclusive lower site index
    hi = istart  # exclusive upper site index (site < istart)
    num_steps = jnp.maximum(hi - lo, 0)

    tensor_shape = (chi_max, p, chi_max)
    matrix_shape = (chi_max, p * chi_max)

    def body_fun(k, As):
        # k = 0, 1, ... maps to i = hi-1, hi-2, ... = lo
        i = hi - 1 - k

        A = As[i]
        C = A.reshape(matrix_shape)

        # RQ decomposition
        R, Q = rq(C)

        A_new = Q.reshape(tensor_shape)
        A_prev_new = jnp.einsum("ijk,kl->ijl", As[i - 1], R)

        As = As.at[i].set(A_new)
        As = As.at[i - 1].set(A_prev_new)
        return As

    # Loop only over the sites that actually need processing
    As = lax.fori_loop(0, num_steps, body_fun, As)
    return As


batched_decreasing_RQ_sweep = jax.vmap(decreasing_RQ_sweep, in_axes=(0, None, None))


# -------------------------- Canonical forms --------------------------

@jax.jit
def canonicalize_to_center(As, center):
    nsites = As.shape[0]
    center = jnp.asarray(center, jnp.int32)

    left_canonical = mps_tensors_are_left_canonical(As)   # (nsites,) bool
    right_canonical = mps_tensors_are_right_canonical(As)  # (nsites,) bool
    idxs = jnp.arange(nsites)

    # ---- Left sweep ----
    mask_left = (~left_canonical) & (idxs < center)
    left_candidates = jnp.where(mask_left, idxs, center)
    qr_start = left_candidates.min()   # JAX scalar int

    # ---- Right sweep ----
    mask_right = (~right_canonical) & (idxs > center)
    right_candidates = jnp.where(mask_right, idxs + 1, center + 1)
    rq_start = right_candidates.max()

    # Sweeps
    As = increasing_QR_sweep(As, istart=qr_start, istop=center)
    As = decreasing_RQ_sweep(As, istart=rq_start, istop=center)

    return As


def ensure_canonical_center_corresponds_to_gate(state, gate_qubits, canonical_center):
    if canonical_center not in gate_qubits:
        state = canonicalize_to_center(state, gate_qubits[0])
    return state


# -------------------------- Other functions --------------------------


def compute_partial_inner_product_from_mps(bra, ket, cut_out_qubits):
    """
    bra, ket: (nsites, chi, p, chi), typically complex
    cut_out_qubits: (2,) JAX array or Python tuple (q1, q2)
    """
    nsites = bra.shape[0]
    cut_out_qubits = jnp.asarray(cut_out_qubits, jnp.int32)
    q1, q2 = cut_out_qubits[0], cut_out_qubits[1]

    dtype = bra.dtype

    # --- Left env (top): contract sites 0 .. q1-1 ---
    Dl_left = bra[0].shape[0]
    top0 = jnp.eye(Dl_left, dtype=dtype)

    def top_body(i, top):
        A1 = ket[i]
        A2 = bra[i]
        return jnp.einsum("ad,abc,dbe->ce", top, A1, A2)

    top = lax.fori_loop(0, q1, top_body, top0)

    # --- Right env (bottom): contract sites nsites-1 .. q2+1 ---
    Dr_right = bra[-1].shape[2]
    bottom0 = jnp.eye(Dr_right, dtype=dtype)

    num_right = nsites - 1 - q2  # number of sites to contract on the right

    def bottom_body(i, bottom):
        site = nsites - 1 - i
        A1 = ket[site]
        A2 = bra[site]
        return jnp.einsum("ce,abc,dbe->ad", bottom, A1, A2)

    bottom = lax.fori_loop(0, num_right, bottom_body, bottom0)

    # --- Contract around q1, q2 ---
    ptr = jnp.einsum(
        "af,abc,cde,fgh,hij,ej->gibd",
        top, ket[q1], ket[q2], bra[q1], bra[q2], bottom
    )

    return ptr


def compute_inner_product_from_mps(psi, phi):
    """
    Given two MPS, compute the inner product = full contraction of outer product.
    No complex conjugation following the convention.

    psi, phi: arrays of shape (nsites, Dl, p, Dr), typically complex.
    """

    # Assume psi, phi already jnp arrays with same shape
    nsites = psi.shape[0]
    Dl_left = psi[0].shape[0]

    # Left boundary: identity on full left bond, with correct dtype
    top0 = jnp.eye(Dl_left, dtype=psi.dtype)  # (Dl_left, Dl_left)

    # Sweep through all sites
    def body(i, top):
        A1 = psi[i]  # (Dl, p, Dr)
        A2 = phi[i]  # (Dl, p, Dr)
        tmp = jnp.einsum("ad,abc->dbc", top, A1)  # (chi, p, chi)
        return jnp.einsum("dbc,dbe->ce", tmp, A2)  # (chi, chi)

    top = lax.fori_loop(0, nsites, body, top0)

    # Right boundary: trace over the right bond
    inner = jnp.einsum("aa->", top)

    return inner


def compute_fidelity(psi, phi):
    overlap = compute_inner_product_from_mps(psi, phi)
    return jnp.abs(overlap)**2


# -------------------------- Functions for unpadded MPS --------------------------

def increasing_SVD_sweep_unpadded(As, max_bondim, rel_tol, abs_tol):
    ''' Assume that the MPS is in right-canonical form! Only for single MPS. '''

    # Left-to-right sweep to compress MPS via SVD
    nsites = len(As)
    for i in range(nsites-1):  # Run through all local tensors
        # Bring local tensor into matrix form
        A = As[i]
        shape = A.shape
        A = A.reshape((-1, shape[-1]))

        # SVD truncation
        u, s, v = svd(A)
        u, s, v = compress_SVD(u, s, v, max_bondim, rel_tol, abs_tol)

        # Restore MPS form
        SV = s[:, None] * v

        As[i] = u.reshape(shape[:-1]+(u.shape[-1],))
        As[i+1] = jnp.einsum('ij,jak->iak', SV, As[i+1])

    return As, s


def decreasing_RQ_sweep_unpadded(As):
    for i in reversed(range(1, len(As))):  # Go from right to left
        # Reshape local tensor to matrix
        A = As[i]
        shape = A.shape
        A = A.reshape((shape[0], -1))

        # Perform RQ
        R, Q = rq(A)

        # Restore MPS form
        A_new = Q.reshape((Q.shape[0],)+shape[1:])
        As[i] = A_new
        As[i-1] = jnp.einsum('ijk,kl->ijl', As[i-1], R)

    return As


def compress_mps(As, max_bondim, rel_tol, abs_tol):
    ''' Non-static bond dimensions. Normalize MPS. input should be unpadded MPS.'''

    # Bring MPS into right-canonical form
    As = decreasing_RQ_sweep_unpadded(As)

    # Compress via SVD
    As, s = increasing_SVD_sweep_unpadded(As, max_bondim, rel_tol, abs_tol)

    # Normalize
    nrm = jnp.sqrt(jnp.sum(s**2))
    As[-1] = As[-1]/nrm
    nrm = jnp.abs(nrm)

    return As
