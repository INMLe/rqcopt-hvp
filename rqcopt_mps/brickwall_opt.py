import os

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from .util import get_tensors
from .objective_function import (
    compute_riemannian_loss_gradient_HST, 
    compute_riemannian_hvp_HST, 
    compute_loss_HST, 
    compute_riemannian_loss_gradient_F, 
    compute_riemannian_hvp_F, 
    compute_loss_F, 
    compute_loss_HST_loop
)
from .riemannian_manifold import (
    retract_to_manifold, 
    project_to_tangent_space, 
    jitted_Hilbert_Schmidt_inner_product
)
from .trust_region import riemannian_trust_region_optimize
from .adam import RieADAM


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

    # ----- Define the objective derivative functions -----
    if cost=='HST':
        func = lambda x: compute_loss_HST(
            psis, phis, x, qubits, config['first_layer_odd'],  config['truncation_dim'], 
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )
        gradfunc = lambda x: compute_riemannian_loss_gradient_HST(
            psis, phis, x, qubits, config['first_layer_odd'], config['truncation_dim'], 
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )
        hessfunc = lambda x,eta: compute_riemannian_hvp_HST(
            psis, phis, x, eta, qubits, config['first_layer_odd'], config['truncation_dim'], 
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )
    elif cost=='F':
        func = lambda x: compute_loss_F(
            psis, phis, x, qubits, config['first_layer_odd'], config['truncation_dim'], 
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )
        gradfunc = lambda x: compute_riemannian_loss_gradient_F(
            psis, phis, x, qubits, config['first_layer_odd'], config['truncation_dim'], 
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )
        hessfunc = lambda x,eta: compute_riemannian_hvp_F(
            psis, phis, x, eta, qubits, config['first_layer_odd'], config['truncation_dim'], 
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )
        
    testfunc = lambda x: compute_loss_HST_loop(
        psis_test, phis_test, x, qubits, config['first_layer_odd'], 
        config['truncation_dim'], config['rel_tol'], config['abs_tol'], config['is_TI']
        )
    

    # ----- Run the optimization -----
    if len(optimizer_params)==1: n_deriv_evals = optimizer_params[0]
    else: n_deriv_evals = 0
    Gs_opt, err_iter, err_test_iter, radius, n_deriv_evals = riemannian_trust_region_optimize(
        func, gradfunc, hessfunc, testfunc, retract_to_manifold, Gs_start,
        niter=config['n_iter'], radius_init=config['radius_init'], 
        maxradius=config['maxradius'], 
        ckpt_dir=config['model_dir'],
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

    # ----- Define the objective derivative functions -----
    if cost=='HST':      
        gradfunc = lambda x: compute_riemannian_loss_gradient_HST(
            psis, phis, x, qubits, config['first_layer_odd'], config['truncation_dim'],
            config['rel_tol'], config['abs_tol'], config['is_TI']
            )
    elif cost=='F':
        gradfunc = lambda x: compute_riemannian_loss_gradient_F(
            psis, phis, x, qubits, config['first_layer_odd'], 
            config['truncation_dim'], config['rel_tol'], config['abs_tol'], 
            config['is_TI']
            )
        
    testfunc = lambda x: compute_loss_HST_loop(
        psis_test, phis_test, x, qubits, config['first_layer_odd'], 
        config['truncation_dim'], config['rel_tol'], config['abs_tol'], config['is_TI']
        )
    
    # ----- Run the optimization -----
    opt = RieADAM(maxiter=config['n_iter'], lr=float(config['lr']), err_iter=err_iter,
                  err_test_iter=err_test_iter,
                  optimizer_params=optimizer_params, ckpt_dir=config['model_dir'])
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
    print("HAHAHA", n_deriv_evals)
    if len(n_deriv_evals)==0: points = jnp.arange(len(err_iter))
    else: points = jnp.cumsum(jnp.asarray(n_deriv_evals)); print('LOL')
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
    plt.semilogy(points[:len(err_iter)], err_iter, '.-', label=label_train)
    plt.semilogy(points[:len(err_test_iter)], err_test_iter, '.-', label=label_test)
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


def test_brickwall_circuit(config, psis_test, phis_test, Gs_opt, qubits, cost='HST',):
    ''' Test the optimized brickwall circuit on test batch. '''

    # Make sure that the gates are in tensor representation
    Gs_opt = get_tensors(Gs_opt)

    # ----- Define the objective derivative functions -----
    if cost=='HST':
        test_loss = compute_loss_HST(
            psis_test, phis_test, Gs_opt, qubits, config['first_layer_odd'],  config['truncation_dim'])
    elif cost=='F':
        test_loss = compute_loss_F(
            psis_test, phis_test, Gs_opt, qubits, config['first_layer_odd'],  config['truncation_dim'])
        
    print(f"Test loss:")
    print(f"----------")
    print(f"err_test = ", test_loss)