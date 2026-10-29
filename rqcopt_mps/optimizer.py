import warnings
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from .riemannian_manifold import jitted_Hilbert_Schmidt_inner_product
from .save_model import (
    make_ckpt_manager, save_ckpt_trust_region, save_ckpt_adam,
)

jax.config.update("jax_enable_x64", True)


# region ========================== Riemannian trust-region ==========================
# This code is taken from https://github.com/qc-tum/rqcopt/blob/master/rqcopt/trust_region.py

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
    rho_trust = kwargs.get("rho_trust", 0.125)
    radius_init = kwargs.get("radius_init", 0.01)
    maxradius = kwargs.get("maxradius",   0.1)
    niter = kwargs.get("niter", 20)
    f_iter = kwargs.get("err_iter", [])
    f_test = kwargs.get("err_test_iter", [])
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
    manager = make_ckpt_manager(ckpt_dir, max_to_keep=kwargs.get('n_ckpt_keep', 10))

    last_step = len(f_iter)
    rho = None  # Agreement quotient of the last step (none before the first step)
    for k in range(last_step, niter+last_step):
        # ----- Compute cost, gradient and HVP function -----
        fx, grad = gradfunc(x)  # Cost and gradient at current iterate
        hess = lambda eta: hessfunc(x, eta)  # HVP function at current iterate
        fx_test = testfunc(x)

        print("\t Step", k)
        if rho is None: print("\t\terror=", fx, "radius=", radius, "test=", fx_test, "tot_inner_iter=", n_deriv_evals[-1])
        else: print("\t\terror=", fx, "radius=", radius, "rho=", rho, "test=", fx_test, "tot_inner_iter=", n_deriv_evals[-1])

        if k == last_step and k > 0:
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
    abstol = kwargs.get("abstol", 1e-8)
    reltol = kwargs.get("reltol", 1e-6)

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

# endregion


# region ========================== Riemannian ADAM ==========================
# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# The present file has been modified to the setting of Riemannian optimization.

class RieADAM():

    def __init__(
        self,
        maxiter: int = 1000,
        tol: float = 1e-9,
        lr: float = 1e-3,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        noise_factor: float = 1e-8,
        eps: float = 1e-10,
        err_iter: list = [],
        err_test_iter: list = [],
        optimizer_params: list = [],
        ckpt_dir: str = '.',
        n_ckpt_keep: int = 10
    ) -> None:
        """
        Args:
            maxiter: Maximum number of iterations
            tol: Tolerance for termination
            lr: Value >= 0, Learning rate.
            beta_1: Value in range 0 to 1, Generally close to 1.
            beta_2: Value in range 0 to 1, Generally close to 1.
            noise_factor: Value >= 0, Noise factor
            eps : Value >=0, Epsilon to be used for finite differences if no analytic
                gradient method is given.
        """
        self._maxiter = maxiter
        self._tol = tol
        self._lr = lr
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._noise_factor = noise_factor
        self._eps = eps
        self._init_t = len(err_iter)
        self._err_iter = err_iter
        self._err_test_iter = err_test_iter
        self._ckpt_dir = ckpt_dir
        self._n_ckpt_keep = n_ckpt_keep  # None keeps all checkpoints

        # runtime variables
        self._t = len(err_iter)
        if len(optimizer_params) == 2: self._m, self._v = optimizer_params
        else: self._m, self._v = 0, 0

    def minimize(
        self,
        function: Callable[[jnp.ndarray], tuple],  # Cost and Riemannian gradient
        testfunction: Callable[[jnp.ndarray], tuple],
        initial_point: jnp.ndarray,
        retract: Callable[[jnp.ndarray], float],  # Retraction
        projection: Callable[[jnp.ndarray], float],  # Projector
        metric: Callable[[jnp.ndarray], float],  # Inner product
    ) -> Tuple[jnp.ndarray, float, int]:

        # ---- Setup checkpoint -----
        manager = make_ckpt_manager(self._ckpt_dir, max_to_keep=self._n_ckpt_keep)

        print('Start Riemannian ADAM optimization ...')

        cost1, derivative = function(initial_point)
        self._err_iter += [cost1]

        cost_test = testfunction(initial_point)
        self._err_test_iter += [cost_test]

        n_out = int(self._maxiter/100) if self._maxiter > 100 else 1
        print('\t', self._t, '\tCurrent cost: ', cost1, '\tTest err: ', cost_test)

        deriv_shape = jnp.shape(derivative)
        if self._init_t == 0 or jnp.sum(self._m) == 0: self._m, self._v = jnp.zeros(deriv_shape), jnp.zeros(deriv_shape[0])

        params = params_new = initial_point
        while self._t < self._maxiter+self._init_t:
            if self._t > self._init_t:
                cost1, derivative = function(params)
                cost_test = testfunction(params)
                if self._t%n_out == 0: print('\t', self._t, '\tCurrent cost: ', cost1, '\tTest err: ', cost_test)
                self._err_iter.append(cost1)
                self._err_test_iter.append(cost_test)

            self._t += 1
            self._m = self._beta_1 * self._m + (1 - self._beta_1) * derivative
            self._v = self._beta_2 * self._v + (1 - self._beta_2) * metric(derivative, derivative)

            lr_eff = self._lr * jnp.sqrt(1 - self._beta_2 ** self._t) / (1 - self._beta_1 ** self._t)
            new_step = - lr_eff * self._m / (jnp.sqrt(self._v) + self._noise_factor)[:, None, None, None, None]
            params_new = retract(params, new_step)  # Apply retraction

            if jnp.linalg.norm(params - params_new) < self._tol:
                # ----- Write one final checkpoint -----
                print("Final checkpoint", self._t, len(self._err_iter))
                save_ckpt_adam(manager, self._t+1, params_new, jnp.asarray(self._err_iter), jnp.asarray(self._err_test_iter), self._m, self._v)
                if jax.process_index() == 0:
                    manager.wait_until_finished()
                    print("latest_step after wait:", manager.latest_step())
                    manager.close()
                return params_new, self._t, self._err_iter, self._err_test_iter, self._m, self._v
            else:
                # ----- Checkpoint -----
                save_ckpt_adam(manager, self._t, params_new, jnp.asarray(self._err_iter), jnp.asarray(self._err_test_iter), self._m, self._v)
                params = params_new

            # Vector transport
            self._m = projection(params_new, jnp.reshape(self._m, jnp.shape(params_new)))

        print('... End Riemannian ADAM optimization')

        # ----- Write one final checkpoint -----
        print("Final checkpoint", self._t, len(self._err_iter))
        save_ckpt_adam(manager, self._t+1, params_new, jnp.asarray(self._err_iter), jnp.asarray(self._err_test_iter), self._m, self._v)
        if jax.process_index() == 0:
            manager.wait_until_finished()
            print("latest_step after wait:", manager.latest_step())
            manager.close()
        return params_new, self._t, self._err_iter, self._err_test_iter, self._m, self._v

# endregion
