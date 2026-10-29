import jax
import jax.numpy as jnp
from jax import config

from .tn_helpers import compress_SVD
from .util import svd, qr

config.update("jax_enable_x64", True)


# region ========================== Matrix-level split ==========================

def _unit_phases(d):
    absd = jnp.abs(d)
    return jnp.where(absd > 0, d / jnp.where(absd > 0, absd, 1.), 1.)


def aug_split_left(theta, D, chi, rel_tol, abs_tol, fill_slack=True):
    """
    Split [[theta, 0], [D, theta]] ~ [[L, 0], [X, L]] @ [[R, 0], [Y, R]]
    with L a left isometry (m x chi) and R = S V^dagger (chi x n).

    Returns L, R, X, Y and err = ||D - (X R + L Y)||_F (the dropped part).
    """
    u, s, vh = svd(theta)
    u, s, vh = compress_SVD(u, s, vh, chi, rel_tol, abs_tol)
    keep = s > 0
    u_r = u * keep[None, :]
    v_r = vh.conj().T * keep[None, :]

    # Part of D outside the tangent space at theta: E = (1 - P_Ur) D (1 - P_Vr)
    D_perp_v = D - (D @ v_r) @ v_r.conj().T
    E = D_perp_v - u_r @ (u_r.conj().T @ D_perp_v)

    if fill_slack:
        # Put the leading left singular vectors of E into the unused columns of u.
        # E is orthogonal to u_r, and QR keeps the first r columns (up to a phase,
        # which is removed again).
        ue, _, _ = svd(E)
        r = jnp.sum(keep)
        candidates = jnp.where(keep[None, :], u, jnp.roll(ue[:, :chi], r, axis=1))
        q, rmat = qr(candidates)
        U = q * _unit_phases(jnp.diagonal(rmat))[None, :]
    else:
        U = u

    s_inv = jnp.where(keep, 1. / jnp.where(keep, s, 1.), 0.)

    L = U
    R = s[:, None] * vh
    Y = U.conj().T @ D                                   # P_U D
    X = (D - U @ Y) @ (v_r * s_inv[None, :])             # (1 - P_U) D V S^+
    err = jnp.linalg.norm(E - U @ (U.conj().T @ E))
    return L, R, X, Y, err


def aug_split_right(theta, D, chi, rel_tol, abs_tol, fill_slack=True):
    """ Mirror of aug_split_left: R is a right isometry, L = U S. """
    L_t, R_t, X_t, Y_t, err = aug_split_left(
        theta.T, D.T, chi, rel_tol, abs_tol, fill_slack
        )
    # theta = R_t^T L_t^T,  D = R_t^T X_t^T + Y_t^T L_t^T
    return R_t.T, L_t.T, Y_t.T, X_t.T, err

# endregion


# region ========================== Two-site update ==========================

def two_site_theta(A_i, A_j, G):
    """ Theta[a,k,l,b] = sum G[k,l,i,j] A_i[a,i,m] A_j[m,j,b] """
    return jnp.einsum('aim,mjb,klij->aklb', A_i, A_j, G)


def two_site_tangent(A_i, A_j, B_i, B_j, G, V):
    """ D = G (B_i A_j + A_i B_j) + V (A_i A_j) """
    return (two_site_theta(B_i, A_j, G) + two_site_theta(A_i, B_j, G)
            + two_site_theta(A_i, A_j, V))


def aug_split_two_site(theta, D, chi, rel_tol, abs_tol, left_canonical, fill_slack=True):
    """
    Split two-site tensors theta, D of shape (vl, p, p, vr) into
    (A_i, A_j, B_i, B_j, err). left_canonical is a static Python bool.
    """
    vl, p1, p2, vr = theta.shape
    th = theta.reshape(vl * p1, p2 * vr)
    Dm = D.reshape(vl * p1, p2 * vr)
    split = aug_split_left if left_canonical else aug_split_right
    L, R, X, Y, err = split(th, Dm, chi, rel_tol, abs_tol, fill_slack)
    return (L.reshape(vl, p1, chi), R.reshape(chi, p2, vr),
            X.reshape(vl, p1, chi), Y.reshape(chi, p2, vr), err)

# endregion


# region ========================== One-site center moves ==========================

def aug_move_right(A_j, B_j, A_next, B_next, rel_tol, abs_tol, fill_slack=True):
    """ Make site j a left isometry and absorb the remainder into site j+1. """
    vl, p, vr = A_j.shape
    L, R, X, Y, err = aug_split_left(
        A_j.reshape(vl * p, vr), B_j.reshape(vl * p, vr), vr, rel_tol, abs_tol, fill_slack
        )
    A_next_new = jnp.einsum('ij,jkl->ikl', R, A_next)
    B_next_new = (jnp.einsum('ij,jkl->ikl', R, B_next)
                  + jnp.einsum('ij,jkl->ikl', Y, A_next))
    return L.reshape(vl, p, vr), X.reshape(vl, p, vr), A_next_new, B_next_new, err


def aug_move_left(A_prev, B_prev, A_i, B_i, rel_tol, abs_tol, fill_slack=True):
    """ Make site i a right isometry and absorb the remainder into site i-1. """
    vl, p, vr = A_i.shape
    L, R, X, Y, err = aug_split_right(
        A_i.reshape(vl, p * vr), B_i.reshape(vl, p * vr), vl, rel_tol, abs_tol, fill_slack
        )
    A_prev_new = jnp.einsum('ijk,kl->ijl', A_prev, L)
    B_prev_new = (jnp.einsum('ijk,kl->ijl', B_prev, L)
                  + jnp.einsum('ijk,kl->ijl', A_prev, X))
    return A_prev_new, B_prev_new, R.reshape(vl, p, vr), Y.reshape(vl, p, vr), err

# endregion


# region ========================== MPS-level gate application ==========================

def _aug_gate_then_move_right(As, Bs, G, V, gate_qubits, chi, rel_tol, abs_tol, fill_slack):
    i, j = gate_qubits[0], gate_qubits[1]
    theta = two_site_theta(As[i], As[j], G)
    D = two_site_tangent(As[i], As[j], Bs[i], Bs[j], G, V)
    A_i, A_j, B_i, B_j, err_split = aug_split_two_site(
        theta, D, chi, rel_tol, abs_tol, left_canonical=True, fill_slack=fill_slack
        )
    # Push the canonical center to site j+1
    A_j, B_j, A_n, B_n, err_move = aug_move_right(
        A_j, B_j, As[j + 1], Bs[j + 1], rel_tol, abs_tol, fill_slack
        )
    As = As.at[i].set(A_i).at[j].set(A_j).at[j + 1].set(A_n)
    Bs = Bs.at[i].set(B_i).at[j].set(B_j).at[j + 1].set(B_n)
    return As, Bs, jnp.int32(j + 1), err_split + err_move


def _aug_gate_then_move_left(As, Bs, G, V, gate_qubits, chi, rel_tol, abs_tol, fill_slack):
    i, j = gate_qubits[0], gate_qubits[1]
    theta = two_site_theta(As[i], As[j], G)
    D = two_site_tangent(As[i], As[j], Bs[i], Bs[j], G, V)
    A_i, A_j, B_i, B_j, err_split = aug_split_two_site(
        theta, D, chi, rel_tol, abs_tol, left_canonical=False, fill_slack=fill_slack
        )
    # Push the canonical center to site i-1
    A_p, B_p, A_i, B_i, err_move = aug_move_left(
        As[i - 1], Bs[i - 1], A_i, B_i, rel_tol, abs_tol, fill_slack
        )
    As = As.at[i - 1].set(A_p).at[i].set(A_i).at[j].set(A_j)
    Bs = Bs.at[i - 1].set(B_p).at[i].set(B_i).at[j].set(B_j)
    return As, Bs, jnp.int32(i - 1), err_split + err_move


def apply_aug_gate(As, Bs, G, V, gate_qubits, left_canonical,
                   truncation_dim, rel_tol, abs_tol, fill_slack=True):
    """
    Apply the augmented gate W = [[G, 0], [V, G]] to (As, Bs).
    The canonical center must be on one of the gate qubits. Sweep direction and
    boundary handling follow tn_methods.merge_gate_with_mps_wrapper.

    Returns As, Bs, the new canonical center and the dropped norm of this step.
    """
    L = As.shape[0]
    q0 = gate_qubits[0]
    is_left_boundary = (q0 == 0)
    is_right_boundary = (q0 == (L - 2))
    use_left = jnp.where(left_canonical, ~is_right_boundary, is_left_boundary)

    args = (As, Bs, G, V, gate_qubits)
    statics = (truncation_dim, rel_tol, abs_tol, fill_slack)
    As, Bs, cc, err = jax.lax.cond(
        use_left,
        lambda a: _aug_gate_then_move_right(*a, *statics),
        lambda a: _aug_gate_then_move_left(*a, *statics),
        args,
    )
    return As, Bs, cc, err

# endregion


# region ========================== Contractions ==========================

def partial_inner_product_tangent_ket(bra, As, Bs, cut_out_qubits):
    """
    Same as mps.compute_partial_inner_product_from_mps(bra, ket, cut) but with the
    ket replaced by the tangent state sum_k A..B_k..A. Uses two-component
    environments (plain, tangent), so the cost is about twice the plain version.

    Output indices: (bra q1, bra q2, ket q1, ket q2), as in the plain version.
    """
    nsites = bra.shape[0]
    cut_out_qubits = jnp.asarray(cut_out_qubits, jnp.int32)
    q1, q2 = cut_out_qubits[0], cut_out_qubits[1]

    step_top = lambda T, K, Bra: jnp.einsum("ad,abc,dbe->ce", T, K, Bra)
    step_bottom = lambda R, K, Bra: jnp.einsum("ce,abc,dbe->ad", R, K, Bra)

    eye_l = jnp.eye(bra.shape[1], dtype=bra.dtype)
    eye_r = jnp.eye(bra.shape[3], dtype=bra.dtype)

    def top_body(i, c):
        T0, T1 = c
        return (step_top(T0, As[i], bra[i]),
                step_top(T1, As[i], bra[i]) + step_top(T0, Bs[i], bra[i]))

    T0, T1 = jax.lax.fori_loop(0, q1, top_body, (eye_l, jnp.zeros_like(eye_l)))

    def bottom_body(i, c):
        R0, R1 = c
        site = nsites - 1 - i
        return (step_bottom(R0, As[site], bra[site]),
                step_bottom(R1, As[site], bra[site]) + step_bottom(R0, Bs[site], bra[site]))

    R0, R1 = jax.lax.fori_loop(0, nsites - 1 - q2, bottom_body, (eye_r, jnp.zeros_like(eye_r)))

    def center(T, K1, K2, R):
        return jnp.einsum("af,abc,cde,fgh,hij,ej->gibd", T, K1, K2, bra[q1], bra[q2], R)

    A1, A2, B1, B2 = As[q1], As[q2], Bs[q1], Bs[q2]
    return (center(T1, A1, A2, R0) + center(T0, B1, A2, R0)
            + center(T0, A1, B2, R0) + center(T0, A1, A2, R1))


def partial_inner_product_tangent_bra(As, Bs, ket, cut_out_qubits):
    """ Tangent state as the bra; same output index order as the plain version. """
    return partial_inner_product_tangent_ket(ket, As, Bs, cut_out_qubits).transpose(2, 3, 0, 1)

# endregion


# region ========================== Conversion ==========================


# endregion
