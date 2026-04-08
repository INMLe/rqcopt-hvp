# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# The present file has been modified to the setting of Riemannian optimization.

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from .save_model import make_ckpt_manager, save_ckpt_adam


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
        ckpt_dir: str = '.'
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

        # runtime variables
        self._t = len(err_iter)  
        if len(optimizer_params)==2: self._m, self._v = optimizer_params
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
        manager = make_ckpt_manager(self._ckpt_dir)

        print('Start Riemannian ADAM optimization ...')

        cost1, derivative = function(initial_point)
        self._err_iter += [cost1]

        cost_test = testfunction(initial_point)
        self._err_test_iter += [cost_test]

        n_out=int(self._maxiter/100) if self._maxiter>100 else 1
        print('\t', self._t, '\tCurrent cost: ', cost1, '\tTest err: ', cost_test)

        deriv_shape = jnp.shape(derivative)
        if self._init_t==0 or jnp.sum(self._m)==0: self._m, self._v = jnp.zeros(deriv_shape), jnp.zeros(deriv_shape[0])

        params = params_new = initial_point
        while self._t < self._maxiter+self._init_t:
            if self._t > self._init_t:
                cost1, derivative = function(params)
                cost_test = testfunction(params)
                if self._t%n_out==0: 
                    print('\t', self._t, '\tCurrent cost: ', cost1, '\tTest err: ', cost_test)
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