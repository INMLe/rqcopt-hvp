import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .util import svd, qr, rq
from .tn_helpers import (
    compress_SVD, 
    right_canonicalize_local_tensor, 
    compress_SVD_padded,
    left_canonicalize_local_tensor, 
    mps_tensors_are_left_canonical, 
    mps_tensors_are_right_canonical,
    )


#region =============== Helper functions for MPS ===============

def maximum_bond_dimension(As):
    return max([dim.shape[2] for dim in As])

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
    A = A[jnp.newaxis,...,jnp.newaxis]  # Extend by dummy bonds

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
        if site==N-1: As.append(F)  
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
    key_real, key_imag = jax.random.split(key)
    real_part = jax.random.uniform(key_real, shape=(2**nsites,))
    imag_part = jax.random.uniform(key_imag, shape=(2**nsites,))
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
    key_real, key_imag = jax.random.split(key)
    # draw real and imag parts in batch
    real_part = jax.random.normal(key_real, shape=(nbatch, dim))
    imag_part = jax.random.normal(key_imag, shape=(nbatch, dim))
    psis = real_part + 1j * imag_part

    # normalize per state
    norms = jnp.linalg.norm(psis, axis=1, keepdims=True)
    psis = psis / norms

    # convert batch of state vectors -> list of stacked local tensors
    As = get_mps_from_vector_stacked(psis)
    return As

def get_haar_random_product_state(key, nsites, d=2):
    """
    Generate a Haar-random product state over N sites with local dimension d.
    
    Parameters:
        key: jax.random.PRNGKey - the random key.
        nsites: int - number of sites.
        d: int - local Hilbert space dimension, d=2 for qubits.
    """
    keys = jax.random.split(key, nsites)
    
    def sample_site(k):
        k_re, k_im = jax.random.split(k, 2)
        psi = jax.random.normal(k_re, (d,)) + 1j * jax.random.normal(k_im, (d,))
        psi /= jnp.linalg.norm(psi)
        return psi.reshape(1, d, 1)
    
    As = [sample_site(k) for k in keys]
    As = decreasing_RQ_sweep(As, istart=nsites, istop=0)
    
    return As

def get_haar_random_product_state_stacked(key, nsites, nsamples, d=2):
    """
    Returns: list length nsites, each element has shape (nsamples, 1, d, 1).
    """
    # independent randomness for real/imag
    k_re, k_im = jax.random.split(key, 2)

    # sample complex Gaussian for every (sample, site, local_dim)
    re = jax.random.normal(k_re, (nsamples, nsites, d))
    im = jax.random.normal(k_im, (nsamples, nsites, d))
    psi = re + 1j * im  # (nsamples, nsites, d)

    # normalize each local state vector (over d)
    psi = psi / jnp.linalg.norm(psi, axis=-1, keepdims=True)

    # reshape to (nsamples, nsites, 1, d, 1)
    psi = psi.reshape(nsamples, nsites, 1, d, 1)

    # return list of length nsites, each (nsamples, 1, d, 1)
    return jnp.asarray([psi[:, i, :, :, :] for i in range(nsites)])

#endregion


#region =============== Local tensors in an MPS ===============

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

batched_left_canonicalize_local_tensor_in_mps = jax.vmap(
    left_canonicalize_local_tensor_in_mps, 
    in_axes=(0, None),
    out_axes=0
)

batched_right_canonicalize_local_tensor_in_mps = jax.vmap(
    right_canonicalize_local_tensor_in_mps, 
    in_axes=(0, None),
    out_axes=0
)

#endregion



#region =============== SWEEPS ===============

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

        return As, s

    As, s_last = jax.lax.fori_loop(0, nsites - 1, body_fun, (As, s0))
    return As, s_last

def _batched_increasing_SVD_sweep_core(As, max_bondim, rel_tol, abs_tol):
    return jax.vmap(
    _increasing_SVD_sweep_core, 
    in_axes=(0, None, None, None), 
    out_axes=(0, 0)
    )(As, max_bondim, rel_tol, abs_tol)

def increasing_SVD_sweep(As, max_bondim, rel_tol, abs_tol):
    return _increasing_SVD_sweep_core(As, max_bondim, rel_tol, abs_tol)[0]

def batched_increasing_SVD_sweep(As, max_bondim, rel_tol, abs_tol):
    return _batched_increasing_SVD_sweep_core(As, max_bondim, rel_tol, abs_tol)[0]

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

    istart = jnp.asarray(istart, jnp.int32)
    istop  = jnp.asarray(istop,  jnp.int32)
    istart = jnp.clip(istart, 0, nsites - 1)
    istop  = jnp.clip(istop,  0, nsites - 1)

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
    As = jax.lax.fori_loop(istart, istop, body_fun, As)
    return As

batched_increasing_QR_sweep = jax.vmap(increasing_QR_sweep, in_axes=(0, None, None))

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
    istop  = jnp.asarray(istop,  jnp.int32)
    istart = jnp.clip(istart, 0, nsites)
    istop  = jnp.clip(istop,  0, nsites - 1)

    # Effective range: sites from istart-1 down to max(istop+1, 1)
    # We also need i > 0 because we access As[i-1].
    lo = jnp.maximum(istop + 1, 1)   # inclusive lower site index
    hi = istart                       # exclusive upper site index (site < istart)
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
    As = jax.lax.fori_loop(0, num_steps, body_fun, As)
    return As

batched_decreasing_RQ_sweep = jax.vmap(decreasing_RQ_sweep, in_axes=(0, None, None))

def reveal_spectra(As, max_bondim, increasing_order=True, rel_tol=1e-16, abs_tol=1e-16):
    """
    Sweeps Left-to-Right through the MPS, canonicalizing it and 
    measuring the singular values at each bond.

    Args:
        As: MPS tensors of shape (nsites, chi, p, chi)
        max_bondim: The static dimension to truncate back to.
        cutoff: Noise threshold (singular values < cutoff set to 0.0)

    Returns:
        As_canonical: The new MPS tensors (Left-Canonical form).
        spectra: The singular values at each bond (nsites-1, max_bondim).
    """
    # Setup shapes
    nsites, chi, p, _ = As.shape
    
    if increasing_order:
        # 'carry' will be the tensor currently holding the orthogonality center.
        init_carry = As[0] 
        xs = As[1:]

        def scan_body(left_tensor, right_tensor):
            # --- Merge ---
            theta = jnp.tensordot(left_tensor, right_tensor, axes=(2, 0))
            
            # --- Reshape for SVD ---
            theta_mat = theta.reshape(chi * p, p * chi)
            
            # --- SVD ---
            U, S, Vh = svd(theta_mat)
            U = U[:, :max_bondim]
            S = S[:max_bondim]
            Vh = Vh[:max_bondim, :]

            # Hybrid threshold
            threshold = jnp.maximum(abs_tol, jnp.max(S) * rel_tol)
            mask = (S > threshold).astype(S.dtype)
            S_clean = S * mask
            
            # --- Split ---
            A_left_new = U.reshape(chi, p, max_bondim)
            
            # New Carry (Orthogonality Center): Contract S @ V into the next site
            SVh = S_clean[:, None] * Vh
            next_carry = SVh.reshape(max_bondim, p, chi)
            
            return next_carry, (A_left_new, S_clean)

        # Run the scan
        final_carry, (stacked_As_left, stacked_spectra) = jax.lax.scan(scan_body, init_carry, xs)
        
        # --- Reassemble the MPS ---
        final_carry = final_carry.reshape(1, chi, p, chi)
        As_canonical = jnp.concatenate([stacked_As_left, final_carry], axis=0)
        
        return As_canonical, stacked_spectra
    
    else:
        # Start with the last site (the current center)
        init_carry = As[-1]
        
        # Iterate backwards over the rest (N-2 down to 0)
        xs = As[:-1][::-1] 

        def scan_body(right_tensor, left_tensor):            
            # --- Merge ---
            theta = jnp.tensordot(left_tensor, right_tensor, axes=(2, 0))
            
            # --- Reshape for SVD ---
            theta_mat = theta.reshape(chi * p, p * chi)
            
            # ---SVD ---
            U, S, Vh = svd(theta_mat)
            U = U[:, :max_bondim]
            S = S[:max_bondim]
            Vh = Vh[:max_bondim, :]

            # Hybrid threshold
            threshold = jnp.maximum(abs_tol, jnp.max(S) * rel_tol)
            mask = (S > threshold).astype(S.dtype)
            S_clean = S * mask
            
            # --- Split ---
            A_right_new = Vh.reshape(max_bondim, p, chi)
            
            # New carry
            US = U * S_clean[None, :]
            next_carry = US.reshape(chi, p, max_bondim)

            return next_carry, (A_right_new, S_clean)

        # Run Scan
        final_carry, (stacked_As_right, stacked_spectra) = jax.lax.scan(scan_body, init_carry, xs)
        
        # Reverse the stacked arrays back to physical order [1, ..., N-1]
        stacked_As_right = stacked_As_right[::-1]
        stacked_spectra = stacked_spectra[::-1]
        
        # Reshape final_carry (site 0)
        final_carry = final_carry.reshape(1, chi, p, chi)
        
        # Concatenate: [site 0] + [site 1...N-1]
        As_canonical = jnp.concatenate([final_carry, stacked_As_right], axis=0)
        
        return As_canonical, stacked_spectra
    
def reveal_central_spectrum(As, max_bondim, rel_tol=1e-16, abs_tol=1e-16):
    """
    Uses QR and RQ sweeps to isolate the central bond and reveal its spectrum.
    """
    nsites = As.shape[0]
    mid = nsites // 2

    # Obtain mixed canonical form
    As_mixed = canonicalize_to_center(As, mid)

    # Extract the central tensor
    center_tensor = As_mixed[mid]

    # SVD of the center tensor to find the bond spectrum
    dim_l, p, dim_r = center_tensor.shape
    matrix_center = center_tensor.reshape(dim_l * p, dim_r)

    # Singular values only
    _, spectrum, _ = svd(matrix_center)

    # Apply cutoff and truncate to max_bondim
    threshold = jnp.maximum(abs_tol, jnp.max(spectrum) * rel_tol)
    spectrum_clean = jnp.where(spectrum > threshold, spectrum, 0.0)
    
    return spectrum_clean[:max_bondim]


#endregion


#region =============== Canonical forms ===============

@jax.jit
def canonicalize_to_center(As, center):
    nsites = As.shape[0]
    center = jnp.asarray(center, jnp.int32)

    left_canonical  = mps_tensors_are_left_canonical(As)   # (nsites,) bool
    right_canonical = mps_tensors_are_right_canonical(As)  # (nsites,) bool
    idxs = jnp.arange(nsites)

    # ---- Left sweep: first non-left-canonical site strictly left of center ----
    mask_left = (~left_canonical) & (idxs < center)
    left_candidates = jnp.where(mask_left, idxs, center)
    qr_start = left_candidates.min()   # JAX scalar int

    # ---- Right sweep: rightmost non-right-canonical site strictly right of center ----
    mask_right = (~right_canonical) & (idxs > center)
    right_candidates = jnp.where(mask_right, idxs + 1, center + 1)
    rq_start = right_candidates.max()  # JAX scalar int

    # Sweeps
    As = increasing_QR_sweep(As, istart=qr_start, istop=center)
    As = decreasing_RQ_sweep(As, istart=rq_start, istop=center)

    return As

@jax.jit
def batched_canonicalize_to_center(As, center):
    return jax.vmap(
    canonicalize_to_center, 
    in_axes=(0, None),
    out_axes=0
)(As, center)

@jax.jit
def canonicalize_from_known_center(As, current_center, target_center):
    """
    Sweep the canonical center from current_center → target_center.
    No canonicality check — caller guarantees where the center currently is.

    Both current_center and target_center should be JAX integer scalars.
    Only one of the two sweeps will do actual work (the other has an
    empty range and becomes a no-op via the internal lax.cond guards).
    """
    # If current < target: QR sweep moves center rightward
    As = increasing_QR_sweep(As, istart=current_center, istop=target_center)
    # If current > target: RQ sweep moves center leftward
    As = decreasing_RQ_sweep(As, istart=current_center + 1, istop=target_center)
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

is_mixed_canonical_batched = jax.vmap(is_mixed_canonical, in_axes=(0, None))
is_left_canonical_batched = jax.vmap(is_left_canonical, in_axes=(0,))
is_right_canonical_batched = jax.vmap(is_right_canonical, in_axes=(0,))

def left_canonicalize(As):
    return increasing_QR_sweep(As, 0, len(As)) #canonicalize_to_center(As, len(As)-1)

def right_canonicalize(As):
    return decreasing_RQ_sweep(As, len(As), 0) #canonicalize_to_center(As, 0)

def batched_left_canonicalize(As):
    return batched_increasing_QR_sweep(As, 0, As.shape[1])

def batched_right_canonicalize(As):
    return batched_decreasing_RQ_sweep(As, As.shape[1], 0)

def ensure_canonical_center_corresponds_to_gate(state, gate_qubits, canonical_center):
    if canonical_center not in gate_qubits: 
        state = canonicalize_to_center(state, gate_qubits[0])
    return state

def batched_ensure_canonical_center_corresponds_to_gate(state, gate_qubits, canonical_center):
    if canonical_center not in gate_qubits: 
        state = batched_canonicalize_to_center(state, gate_qubits[0])
    return state

#endregion



#region =============== Other functions ===============

def compute_partial_inner_product_from_mps(bra, ket, cut_out_qubits):
    """
    bra, ket: (nsites, chi, p, chi), typically complex
    cut_out_qubits: (2,) JAX array or Python tuple (q1, q2)
    """
    nsites = bra.shape[0]
    cut_out_qubits = jnp.asarray(cut_out_qubits, jnp.int32)
    q1, q2 = cut_out_qubits[0], cut_out_qubits[1]

    dtype = bra.dtype

    # --- Left env (top): contract sites 0 ... q1-1 ---
    Dl_left = bra[0].shape[0]
    top0 = jnp.eye(Dl_left, dtype=dtype)

    def top_body(i, top):
        A1 = ket[i]
        A2 = bra[i]
        return jnp.einsum("ad,abc,dbe->ce", top, A1, A2)

    top = jax.lax.fori_loop(0, q1, top_body, top0)

    # --- Right env (bottom): contract sites nsites-1 ... q2+1 ---
    Dr_right = bra[-1].shape[2]
    bottom0 = jnp.eye(Dr_right, dtype=dtype)

    num_right = nsites - 1 - q2  # number of sites to contract on the right

    def bottom_body(i, bottom):
        site = nsites - 1 - i
        A1 = ket[site]
        A2 = bra[site]
        return jnp.einsum("ce,abc,dbe->ad", bottom, A1, A2)

    bottom = jax.lax.fori_loop(0, num_right, bottom_body, bottom0)

    # --- Contract around q1, q2 ---
    ptr = jnp.einsum(
        "af,abc,cde,fgh,hij,ej->gibd",
        top, ket[q1], ket[q2], bra[q1], bra[q2], bottom
    )

    return ptr

@jax.jit
def jitted_compute_partial_inner_product_from_mps(bra, ket, cut_out_qubits):
    return compute_partial_inner_product_from_mps(bra, ket, cut_out_qubits)

@jax.jit
def batched_compute_partial_inner_product_from_mps(bras, kets, cut_out_qubits):
    return jax.vmap(compute_partial_inner_product_from_mps, in_axes=(0, 0, None))(bras, kets, cut_out_qubits)

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

    # Sweep through all sites with a fori_loop (JAX-friendly)
    def body(i, top):
        A1 = psi[i]  # (Dl, p, Dr)
        A2 = phi[i]  # (Dl, p, Dr)
        tmp = jnp.einsum("ad,abc->dbc", top, A1)   # (chi, p, chi)
        return jnp.einsum("dbc,dbe->ce", tmp, A2)   # (chi, chi)

    top = jax.lax.fori_loop(0, nsites, body, top0)

    # Right boundary: trace over the right bond
    inner = jnp.einsum("aa->", top)

    return inner

@jax.jit
def jitted_compute_inner_product_from_mps(psi, phi):
    return compute_inner_product_from_mps(psi, phi)

@jax.jit
def batched_compute_inner_product_from_mps(psi, phi):
    return jax.vmap(compute_inner_product_from_mps, in_axes=(0, 0))(psi, phi)

def compute_fidelity(psi, phi):
    overlap = compute_inner_product_from_mps(psi, phi)
    return jnp.abs(overlap)**2

batched_compute_fidelity = jax.vmap(
    compute_fidelity, in_axes=(0, 0), out_axes=0)

def compute_vector_norm(psi, phi):
    nrm = 2*(1 - compute_inner_product_from_mps(psi,phi).real)
    return jnp.sqrt(nrm)

#endregion



#region =============== Functions for unpadded MPS ===============

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

#endregion