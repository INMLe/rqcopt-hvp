from functools import partial

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from .tn_helpers import (
    truncate_and_right_canonicalize_local_tensor, 
    truncate_and_left_canonicalize_local_tensor, 
    build_block_diagonal_tensor
)
from .mps import (
    batched_canonicalize_to_center, 
    canonicalize_from_known_center
)
from .util import mask_to_zeros

# --- Contraction Helpers ---
def contract_carry_decreasing(A_block, carry):
    """Contracts carry from Right to Left."""
    return jnp.einsum('abc,cd->abd', A_block, carry)

def contract_carry_increasing(A_block, carry):
    """Contracts carry from Left to Right."""
    return jnp.einsum('jkl,ij->ikl', A_block, carry)

# --- Site Processing Helpers ---
def process_first_site_at_start(A0_0, A0_1, truncation_dim, rel_tol, abs_tol):
    """Handles the first site when starting a sweep."""
    A_first = jnp.concatenate([A0_0, A0_1], axis=2)
    A_first, carry = truncate_and_left_canonicalize_local_tensor(
        A_first, truncation_dim, rel_tol, abs_tol
        )
    return A_first, carry

def process_first_site_at_end(A0_0, A0_1, carry):
    """Handles the first site when ending a sweep (Right -> Left)."""
    A_first = jnp.concatenate([A0_0, A0_1], axis=2)
    A_first = contract_carry_decreasing(A_first, carry)
    return A_first

def process_last_site_at_start(A_last_0, A_last_1, truncation_dim, rel_tol, abs_tol):
    """Handles the last site when starting a sweep (Right -> Left)."""
    A_last = jnp.concatenate([A_last_0, A_last_1], axis=0)
    A_last, carry = truncate_and_right_canonicalize_local_tensor(
        A_last, truncation_dim, rel_tol, abs_tol
        )
    return A_last, carry

def process_last_site_at_end(A_last_0, A_last_1, carry):
    """Handles the last site when ending a sweep (Left -> Right)."""
    A_last = jnp.concatenate([A_last_0, A_last_1], axis=0)
    A_last = contract_carry_increasing(A_last, carry)
    return A_last

def _step_in_middle_sweep_increasing(A0_i, A1_i, carry, truncation_dim, rel_tol, abs_tol):
    A_block = build_block_diagonal_tensor(A0_i, A1_i)
    A_block = contract_carry_increasing(A_block, carry)
    # Returns local_loss from compress_SVD inside truncate_...
    A_i, carry_new = truncate_and_left_canonicalize_local_tensor(
        A_block, truncation_dim, rel_tol, abs_tol)
    return A_i, carry_new

def process_middle_sweep_increasing(result_As, mps0, mps1, istart, istop, 
                                    carry, truncation_dim, rel_tol, abs_tol):
    # state = (result_As, carry, total_loss)
    def body(i, state):
        res_As, cry = state
        A_i, cry_new = _step_in_middle_sweep_increasing(
            mps0[i], mps1[i], cry, truncation_dim, rel_tol, abs_tol
            )
        res_As = res_As.at[i].set(A_i)
        return (res_As, cry_new)

    result_As, carry = jax.lax.fori_loop(istart, istop, body, (result_As, carry))
    return result_As, carry

def _step_in_process_middle_sweep_decreasing(A0_i, A1_i, carry, truncation_dim, rel_tol, abs_tol):
    A_block = build_block_diagonal_tensor(A0_i, A1_i)
    A_block = contract_carry_decreasing(A_block, carry)
    A_i, carry_new = truncate_and_right_canonicalize_local_tensor(
        A_block, truncation_dim, rel_tol, abs_tol
        )
    return A_i, carry_new

def process_middle_sweep_decreasing(result_As, mps0, mps1, istart, istop, carry, truncation_dim, rel_tol, abs_tol):
    def body(k, state):
        res_As, cry = state
        i = istop - 1 - k  # reverse mapping: k=0 -> istop-1, k=1 -> istop-2, ...
        A_i, cry_new = _step_in_process_middle_sweep_decreasing(
            mps0[i], mps1[i], cry, truncation_dim, rel_tol, abs_tol
            )
        res_As = res_As.at[i].set(A_i)
        return (res_As, cry_new)

    result_As, carry = jax.lax.fori_loop(0, istop - istart, body, (result_As, carry))
    return result_As, carry

def _solve_center_at_start(mps0, mps1, truncation_dim, rel_tol, abs_tol):
    nsites = mps0.shape[0]
    p = mps0[0].shape[1]
    result_As = jnp.empty((nsites, truncation_dim, p, truncation_dim), dtype=mps0[0].dtype)
    
    # Start site loss
    first_As, carry = process_first_site_at_start(mps0[0], mps1[0], truncation_dim, rel_tol, abs_tol)
    result_As = result_As.at[0].set(first_As)
    
    # Middle sweep loss
    result_As, carry = process_middle_sweep_increasing(
        result_As, mps0, mps1, 1, nsites-1, carry, truncation_dim, rel_tol, abs_tol
        )
    
    # Last site (No truncation loss here as it's just a contraction)
    result_As = result_As.at[-1].set(process_last_site_at_end(mps0[-1], mps1[-1], carry))
    
    return result_As

@partial(jax.jit, static_argnames=('truncation_dim','rel_tol','abs_tol'))
def _batched_solve_center_at_start(mps0, mps1, truncation_dim, rel_tol, abs_tol):
    return jax.vmap(_solve_center_at_start, in_axes=(0, 0, None, None, None), out_axes=0)(
        mps0, mps1, truncation_dim, rel_tol, abs_tol
        )

@partial(jax.jit, static_argnames=('truncation_dim', 'rel_tol', 'abs_tol'))
def _jitted_solve_center_at_start(mps0, mps1, truncation_dim, rel_tol, abs_tol):
    return _solve_center_at_start(mps0, mps1, truncation_dim, rel_tol, abs_tol)

def _solve_center_at_end(mps0, mps1, truncation_dim, rel_tol, abs_tol):
    nsites = mps0.shape[0]
    p = mps0[0].shape[1]
    result_As = jnp.empty((nsites, truncation_dim, p, truncation_dim), dtype=mps0[0].dtype)

    # Start at last site
    A_last, carry = process_last_site_at_start(mps0[-1], mps1[-1], truncation_dim, rel_tol, abs_tol)
    result_As = result_As.at[-1].set(A_last)

    # Sweep backwards
    result_As, carry = process_middle_sweep_decreasing(
        result_As, mps0, mps1, 1, nsites-1, carry, truncation_dim, rel_tol, abs_tol
        )

    # End at first site
    result_As = result_As.at[0].set(process_first_site_at_end(mps0[0], mps1[0], carry))
    
    return result_As

@partial(jax.jit, static_argnames=('truncation_dim', 'rel_tol', 'abs_tol'))
def _jitted_solve_center_at_end(mps0, mps1, truncation_dim, rel_tol, abs_tol):
    return _solve_center_at_end(mps0, mps1, truncation_dim, rel_tol, abs_tol)

@partial(jax.jit, static_argnames=('truncation_dim','rel_tol', 'abs_tol'))
def _batched_solve_center_at_end(mps0, mps1, truncation_dim, rel_tol, abs_tol):
    return jax.vmap(_solve_center_at_end, in_axes=(0, 0, None, None, None), out_axes=0)(
        mps0, mps1, truncation_dim, rel_tol, abs_tol
        )

@partial(jax.jit, static_argnames=("truncation_dim", "rel_tol", "abs_tol"))
def batched_add_mps(mps0, mps1, canonical_center, truncation_dim, rel_tol, abs_tol):
    """
    Adds two MPS states: result = mps0 + mps1
    with immediate bond truncation after each site.

    Important: we always need to add from one edge to the other!
    After this the resulting MPS will be in left or right canonical form (not mixed!)
    """

    L = mps0.shape[1]
    nbatch = mps0.shape[0]
    cc = canonical_center

    def solve_start(m0, m1):
        return _batched_solve_center_at_start(m0, m1, truncation_dim, rel_tol, abs_tol)

    def solve_end(m0, m1):
        return _batched_solve_center_at_end(m0, m1, truncation_dim, rel_tol, abs_tol)

    def to_start_then_solve(m0, m1):
        mps_combined = jnp.concatenate([m0, m1], axis=0)
        mps_combined = batched_canonicalize_to_center(mps_combined, 0)
        m0s, m1s = mps_combined[:nbatch], mps_combined[nbatch:]
        return solve_start(m0s, m1s)

    def to_end_then_solve(m0, m1):
        mps_combined = jnp.concatenate([m0, m1], axis=0)
        mps_combined = batched_canonicalize_to_center(mps_combined, L - 1)
        m0e, m1e = mps_combined[:nbatch], mps_combined[nbatch:]
        return solve_end(m0e, m1e)

    is_start = (cc == 0)
    is_end   = (cc == (L - 1))
    use_end  = ((L - 1 - cc) <= cc)

    # case: 0=start, 1=end, 2=middle->end, 3=middle->start
    case = jnp.where(
        is_start,
        jnp.int32(0),
        jnp.where(is_end, jnp.int32(1), jnp.where(use_end, jnp.int32(2), jnp.int32(3))),
    )

    def branch0(_): return solve_start(mps0, mps1)
    def branch1(_): return solve_end(mps0, mps1)
    def branch2(_): return to_end_then_solve(mps0, mps1)
    def branch3(_): return to_start_then_solve(mps0, mps1)

    result = jax.lax.switch(case, [branch0, branch1, branch2, branch3], operand=None)
    result = mask_to_zeros(result, threshold=1e-16)
    return result

@partial(jax.jit, static_argnames=("truncation_dim","rel_tol", "abs_tol"))
def add_mps(mps0, mps1, canonical_center, truncation_dim, rel_tol, abs_tol):
    """
    Adds two MPS states: result = mps0 + mps1
    with immediate bond truncation after each site.

    Important: we always need to add from one edge to the other!
    After this the resulting MPS will be in left or right canonical form (not mixed!)

    Returns:
        (result, result_cc): the summed MPS and its canonical center
            - case start / middle->start: left-canonical, result_cc = L-1
            - case end / middle->end:     right-canonical, result_cc = 0
    """
    L = mps0.shape[0]
    cc = canonical_center  # jnp scalar integer

    # Decide direction: sweep toward whichever edge is closer.
    # canonicalize_from_known_center is a no-op when cc already equals the target.
    use_end = (L - 1 - cc) <= cc

    def sweep_toward_end(_):
        mps0e = canonicalize_from_known_center(mps0, cc, L - 1)
        mps1e = canonicalize_from_known_center(mps1, cc, L - 1)
        return _jitted_solve_center_at_end(mps0e, mps1e, truncation_dim, rel_tol, abs_tol), jnp.int32(0)

    def sweep_toward_start(_):
        mps0s = canonicalize_from_known_center(mps0, cc, 0)
        mps1s = canonicalize_from_known_center(mps1, cc, 0)
        return _jitted_solve_center_at_start(mps0s, mps1s, truncation_dim, rel_tol, abs_tol), jnp.int32(L - 1)

    result, result_cc = jax.lax.cond(use_end, sweep_toward_end, sweep_toward_start, operand=None)

    result = mask_to_zeros(result, threshold=1e-16)
    return result, result_cc