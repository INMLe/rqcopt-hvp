import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .brickwall_circuit import get_circuit_layout
from .objective_function import (
    compute_riemannian_loss_gradient_HST,
    compute_riemannian_hvp_HST,
    compute_loss_HST,
    compute_riemannian_loss_gradient_F,
    compute_riemannian_hvp_F,
    compute_loss_F,
    compute_loss_HST_loop,
)
from .optimizer import riemannian_trust_region_optimize, RieADAM
from .riemannian_manifold import (
    retract_to_manifold,
    project_to_tangent_space,
    jitted_Hilbert_Schmidt_inner_product,
)
from .util import get_tensors

jax.config.update("jax_enable_x64", True)


def postprocessing_loss(err_iter, err_test_iter, verbose=True):
    err_iter = jnp.asarray(err_iter)  # Loss curve
    err_init = err_iter[0]
    err_opt = jnp.min(err_iter)
    err_end = err_iter[-1]
    err_rel = err_init/err_opt

    err_test_iter = jnp.asarray(err_test_iter)
    err_init_test = err_test_iter[0]
    err_opt_test = jnp.min(err_test_iter)
    err_end_test = err_test_iter[-1]
    err_rel_test = err_init_test/err_opt_test

    if verbose:
        print(f"Training losses:")
        print(f"----------------")
        print(f"err_init: {err_init}")
        print(f"err_end after {len(err_iter)} iterations: {err_end}")
        print(f"err_opt: {err_opt}")
        print(f"err_init/err_opt: {err_rel}")

        print(f"\nTest loss:")
        print(f"----------------")
        print(f"err_init: {err_init_test}")
        print(f"err_end after {len(err_test_iter)} iterations: {err_end_test}")
        print(f"err_opt: {err_opt_test}")
        print(f"err_init/err_opt: {err_rel_test}")

    return err_iter, err_init, err_opt, err_end, err_test_iter, err_init_test, err_opt_test, err_end_test


def optimize_brickwall_circuit_TrustRegion(
        config, psis_all, phis_all, Gs_start, qubits, cost='HST',
        err_iter=[], err_test_iter=[], optimizer_params=[]
        ):
    ''' Optimize the quantum gates in a brickwall circuit to
     approximate the reference unitary U. Cost either HST or F'''

    psis = psis_all[:config['n_samples_train']]
    phis = phis_all[:config['n_samples_train']]

    psis_test = psis_all[-config['n_samples_test']:]
    phis_test = phis_all[-config['n_samples_test']:]

    # Make sure that the gates are in tensor representation
    Gs_start = get_tensors(Gs_start)

    # Gate order of the forward and backward passes (computed once)
    layout = get_circuit_layout(qubits)

    # ----- Define the objective derivative functions -----
    if cost == 'HST':
        func = lambda x: compute_loss_HST(
            psis, phis, x, layout, config['truncation_dim'],
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )
        gradfunc = lambda x: compute_riemannian_loss_gradient_HST(
            psis, phis, x, layout, config['truncation_dim'],
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )
        hessfunc = lambda x, eta: compute_riemannian_hvp_HST(
            psis, phis, x, eta, layout, config['truncation_dim'],
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )
    elif cost == 'F':
        func = lambda x: compute_loss_F(
            psis, phis, x, layout, config['truncation_dim'],
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )
        gradfunc = lambda x: compute_riemannian_loss_gradient_F(
            psis, phis, x, layout, config['truncation_dim'],
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )
        hessfunc = lambda x, eta: compute_riemannian_hvp_F(
            psis, phis, x, eta, layout, config['truncation_dim'],
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )

    testfunc = lambda x: compute_loss_HST_loop(
        psis_test, phis_test, x, layout,
        config['truncation_dim'], config['rel_tol'], config['abs_tol'], config['is_TI']
        )

    # ----- Run the optimization -----
    if len(optimizer_params) == 1: n_deriv_evals = optimizer_params[0]
    else: n_deriv_evals = 0
    Gs_opt, err_iter, err_test_iter, radius, n_deriv_evals = riemannian_trust_region_optimize(
        func, gradfunc, hessfunc, testfunc, retract_to_manifold, Gs_start,
        niter=config['n_iter'], radius_init=config['radius_init'],
        maxradius=config['maxradius'],
        ckpt_dir=config['model_dir'],
        n_ckpt_keep=config.get('n_ckpt_keep', 10),
        err_iter=err_iter,
        err_test_iter=err_test_iter,
        n_deriv_evals=n_deriv_evals
        )

    print("Number of HVP evaluations: ", n_deriv_evals)

    # ----- Postprocessing the loss -----
    err_iter, _, _, _, err_test_iter, _, _, _ = postprocessing_loss(err_iter, err_test_iter, verbose=True)

    _ = plot_loss(config, err_iter, err_test_iter, n_deriv_evals, save=True)

    return Gs_opt, err_iter, err_test_iter, radius, n_deriv_evals


def optimize_brickwall_circuit_ADAM(
        config, psis_all, phis_all, Gs_start, qubits, cost='HST',
        err_iter=[], err_test_iter=[], optimizer_params=[]
        ):
    ''' Optimize the quantum gates in a brickwall circuit to
     approximate the reference unitary U. Cost either HST or F'''

    # Split training and test batch
    psis = psis_all[:config['n_samples_train']]
    phis = phis_all[:config['n_samples_train']]
    psis_test = psis_all[-config['n_samples_test']:]
    phis_test = phis_all[-config['n_samples_test']:]

    # Make sure that the gates are in tensor representation
    Gs_start = get_tensors(Gs_start)

    # Gate order of the forward and backward passes (computed once)
    layout = get_circuit_layout(qubits)

    # ----- Define the objective derivative functions -----
    if cost == 'HST':
        gradfunc = lambda x: compute_riemannian_loss_gradient_HST(
            psis, phis, x, layout, config['truncation_dim'],
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )
    elif cost == 'F':
        gradfunc = lambda x: compute_riemannian_loss_gradient_F(
            psis, phis, x, layout,
            config['truncation_dim'], config['rel_tol'], config['abs_tol'],
            config['is_TI']
            )

    testfunc = lambda x: compute_loss_HST_loop(
        psis_test, phis_test, x, layout,
        config['truncation_dim'], config['rel_tol'], config['abs_tol'], config['is_TI']
        )

    # ----- Run the optimization -----
    opt = RieADAM(maxiter=config['n_iter'], lr=float(config['lr']), err_iter=err_iter,
                  err_test_iter=err_test_iter,
                  optimizer_params=optimizer_params, ckpt_dir=config['model_dir'],
                  n_ckpt_keep=config.get('n_ckpt_keep', 10))
    Gs_opt, neval, err_iter, err_test_iter, m, v = opt.minimize(
        function=gradfunc,
        testfunction=testfunc,
        initial_point=Gs_start,
        retract=retract_to_manifold,
        projection=project_to_tangent_space,
        metric=jitted_Hilbert_Schmidt_inner_product
        )

    # ----- Postprocessing the loss -----
    err_iter, _, _, _, err_test_iter, _, _, _ = postprocessing_loss(err_iter, err_test_iter, verbose=True)

    _ = plot_loss(config, err_iter, err_test_iter, save=True)

    return Gs_opt, err_iter, err_test_iter, m, v


def plot_loss(config, err_iter, err_test_iter, n_deriv_evals=[], mode='final', save=False):

    err_opt = jnp.min(err_iter)
    err_opt_test = jnp.min(err_test_iter)
    err_init = err_iter[0]
    err_init_test = err_test_iter[0]

    # Visualize optimization progress
    # Cumulative number of HVP evaluations after each iteration (trust-region only)
    if len(n_deriv_evals) == 0: points = jnp.array([])
    else: points = jnp.cumsum(jnp.asarray(n_deriv_evals))
    err_iter = jnp.asarray(err_iter)
    err_test_iter = jnp.asarray(err_test_iter)

    label_train = 'err_init={:.2e}\nerr_end={:.2e}\nerr_opt={:.2e}\nerr_init/err_opt={:.4f}'.format(
        err_init, err_iter[-1], err_opt, err_init/err_opt)
    label_test = 'err_init={:.2e}\nerr_end={:.2e}\nerr_opt={:.2e}\nerr_init/err_opt={:.4f}'.format(
        err_init_test, err_test_iter[-1], err_opt_test, err_init_test/err_opt_test)
    title = f"{config['optimizer']} for {config['n_sites']} sites, $t=${config['t']}"
    title += ', '+str(config['n_repetitions']) + ' repetitions '
    title += '(' + mode + ')'

    plt.figure(dpi=300)
    plt.semilogy(err_iter, '.-', label=label_train)
    plt.semilogy(err_test_iter, '.-', label=label_test)
    if len(points) > 0: plt.semilogy(points, err_test_iter, '--', label="HVP evaluations")
    plt.xlabel("Iteration")
    plt.ylabel("$\mathcal{C}$")
    plt.legend()
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()

    if save:
        fname = str(config['model_nbr']) + '_loss_' + mode + '.pdf'
        fdir = os.path.join(config['model_dir'], fname)
        plt.savefig(fdir)
