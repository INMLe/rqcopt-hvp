from time import time
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import warnings


from .riemannian_manifold import jitted_Hilbert_Schmidt_inner_product
from .save_model import make_ckpt_manager, save_ckpt_trust_region

# This code is based on https://github.com/qc-tum/rqcopt/blob/master/rqcopt/trust_region.py


def riemannian_trust_region_optimize(f, gradfunc, hessfunc, testfunc, retract, x_init, **kwargs):
    """
    Optimization via the Riemannian trust-region (RTR) algorithm.

    Reference:
        Algorithm 10 in:
        P.-A. Absil, R. Mahony, Rodolphe Sepulchre
        Optimization Algorithms on Matrix Manifolds
        Princeton University Press (2008)

    f returns function value, derivative, hvp
    """
    rho_trust   = kwargs.get("rho_trust", 0.125)
    radius_init = kwargs.get("radius_init", 0.01)
    maxradius   = kwargs.get("maxradius",   0.1)
    niter       = kwargs.get("niter", 20)
    f_iter      = kwargs.get("err_iter", [])
    f_test      = kwargs.get("err_test_iter", [])
    n_deriv_evals = kwargs.get("n_deriv_evals", [0])


    # transfer keyword arguments for truncated_cg
    tcg_kwargs = {}
    for key in ["maxiter", "abstol", "reltol"]:
        if ("tcg_" + key) in kwargs.keys():
            tcg_kwargs[key] = kwargs["tcg_" + key]
    assert 0 <= rho_trust < 0.25

    # --- Initialize ---
    x = x_init
    radius = radius_init

    # ---- Setup checkpoint -----
    ckpt_dir = kwargs['ckpt_dir']
    manager = make_ckpt_manager(ckpt_dir)

    last_step = len(f_iter)
    for k in range(last_step, niter+last_step):
        # ----- Compute cost, gradient and HVP function -----
        fx, grad = gradfunc(x)  # Cost and gradient at current iterate
        hess = lambda eta: hessfunc(x,eta)  # HVP function at current iterate
        fx_test = testfunc(x)

        print("\t Step", k)
        try: print("\t\terror=", fx, "radius=", radius, "rho=", rho, "test=", fx_test, "tot_inner_iter=", n_deriv_evals[-1])
        except: print("\t\terror=", fx, "radius=", radius, "test=", fx_test, "tot_inner_iter=", n_deriv_evals[-1])

        if k==last_step and k>0: 
            pass
        else:
            f_iter.append(fx)
            f_test.append(fx_test)
            # ----- Checkpoint -----
            save_ckpt_trust_region(manager, k, x, jnp.asarray(f_iter), jnp.asarray(f_test), radius, jnp.asarray(n_deriv_evals))
        
    
        # ----- Solve the trust-region subproblem (Solve Eq. (7.6)) -----
        eta, on_boundary, inner_iterations = truncated_cg(grad, hess, radius, **tcg_kwargs)
        n_deriv_evals.append(inner_iterations)

        # ----- Obtain the agreement quotient -----
        x_next = retract(x, eta)
        rho_nom = f(x_next) - fx

        rho_denom_grad = jitted_Hilbert_Schmidt_inner_product(grad, eta) 
        rho_denom_hess = jitted_Hilbert_Schmidt_inner_product(eta, hess(eta))
        rho_denom = (rho_denom_grad + 0.5*rho_denom_hess).real  # Should be real
        rho = rho_nom / rho_denom  # Eq. (7.7)

        if rho_nom > 0. and rho_denom > 0.:  # Loss did not decrease
            print("Early stopping since loss does not decrease. You should increase the bond dimension.")
            # ----- Write one final checkpoint -----
            print("Final checkpoint", k, len(f_iter))
            save_ckpt_trust_region(manager, k+1, x, jnp.asarray(f_iter), jnp.asarray(f_test), radius, jnp.asarray(n_deriv_evals))
            if jax.process_index() == 0:
                manager.wait_until_finished()
                print("latest_step after wait:", manager.latest_step())
                manager.close()

            return x, f_iter, f_test, radius, n_deriv_evals
        
        
        # ----- Test the agreement quotient -----
        if rho < 0.25:
            # reduce radius
            radius *= 0.25
        elif rho > 0.75 and on_boundary:
            # enlarge radius
            radius = min(2 * radius, maxradius)
        if rho > rho_trust:
            x = x_next

    f_iter.append(f(x))
    f_test.append(testfunc(x))

    # ----- If you finish normally, write one final checkpoint -----
    print("Final checkpoint", k, len(f_iter))
    save_ckpt_trust_region(manager, k+1, x, jnp.asarray(f_iter), jnp.asarray(f_test), radius, jnp.asarray(n_deriv_evals))
    if jax.process_index() == 0:
        manager.wait_until_finished()
        print("latest_step after wait:", manager.latest_step())
        manager.close()

    
    return x, f_iter, f_test, radius, n_deriv_evals

def truncated_cg(grad, hess, radius, **kwargs):
    """
    Truncated CG (tCG) method for the trust-region subproblem:
        minimize   <grad, z> + 1/2 <z, H z>
        subject to <z, z> <= radius^2

    References:
      - Algorithm 11 in:
        P.-A. Absil, R. Mahony, Rodolphe Sepulchre
        Optimization Algorithms on Matrix Manifolds
        Princeton University Press (2008)
      - Trond Steihaug
        The conjugate gradient method and trust regions in large scale optimization
        SIAM Journal on Numerical Analysis 20, 626-637 (1983)
    """

    # grad is numerical value
    # hess is HVP function 
    # Note: here, we take the inner product between two tangent vectors which is real 
    # (we can additionally take the real part for more stable computation)


    maxiter = kwargs.get("maxiter", 2 * len(grad))
    abstol  = kwargs.get("abstol", 1e-8)
    reltol  = kwargs.get("reltol", 1e-6)


    r = grad.copy()  # r0 = grad f(xk)
    z = jnp.zeros_like(r)  # eta0 = 0
    d = -r  # delta0 = -r0

    rsq = jitted_Hilbert_Schmidt_inner_product(r, r)  # Should be real
    stoptol = max(abstol, reltol * jnp.sqrt(rsq))
    
    for j in range(maxiter):
        # ----- Compute $\alpha_j$ -----
        Hd = hess(d)
        dHd = jitted_Hilbert_Schmidt_inner_product(d, Hd)
        alpha = rsq / dHd
        
        # ---- Check for negative curvature (dHd<=0) and boundary hit (alpha>t) ----
        t = _move_to_boundary(z, d, radius)
        if dHd <= 0 or alpha > t:
            # return with move to boundary
            return z + t*d, True, j+1
        
        # update iterates
        r += alpha * Hd  # line 13
        z += alpha * d  # line 8

        # ---- early stopping ----
        rsq_next = jitted_Hilbert_Schmidt_inner_product(r, r)
        if jnp.sqrt(rsq_next) <= stoptol:
            
            return z, False, j+1
        
        beta = rsq_next / rsq
        d = -r + beta * d
        rsq = rsq_next
    # maxiter reached
    return z, False, j+1

def _move_to_boundary(b, d, radius):
    """
    Move to the unit ball boundary by solving
    || b + t*d || == radius
    for t with t > 0.
    """
    dsq = jitted_Hilbert_Schmidt_inner_product(d, d)
    if dsq == 0:
        warnings.warn("input vector 'd' is zero")
        return b
    p = jitted_Hilbert_Schmidt_inner_product(b, d).real / dsq
    q = (jitted_Hilbert_Schmidt_inner_product(b, b).real - radius**2) / dsq
    t = solve_quadratic_equation(p, q)[1]
    if t < 0:
        warnings.warn("encountered t < 0")
    return t

def solve_quadratic_equation(p, q):
    """
    Compute the two solutions of the quadratic equation x^2 + 2 p x + q == 0.
    """
    if p**2 - q < 0:
        print("Negative value", p**2 - q)
        raise ValueError("require non-negative discriminant")
    if p == 0:
        x = jnp.sqrt(-q)
        return (-x, x)
    x1 = -(p + jnp.sign(p)*jnp.sqrt(p**2 - q))
    x2 = q / x1
    return tuple(sorted((x1, x2)))