import os
from sys import argv

import jax.numpy as jnp
import matplotlib.pyplot as plt

from rqcopt_mps import save_model

'''
Use like this: python3 plot-results.py 'ising-1d' 50 (hamiltonian, n_sites)
'''

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")


def plot_optimized_circuit(hamiltonian, n_sites):

    f_results = f'{n_sites}_{hamiltonian}_results.h5'
    f_results = os.path.join(data_dir, f_results)
    f_TEBD = f'{n_sites}_{hamiltonian}_TEBD.h5'
    f_TEBD = os.path.join(data_dir, f_TEBD)

    results = save_model.load_results_h5(f_results)
    TEBD = save_model.load_results_h5(f_TEBD)

    tr_nlayers = list(results['tr_nlayers'])
    tr_test_errors = list(results['tr_test_errors'])
    n_layers_I = list(TEBD['n_layers_I'])
    n_layers_II = list(TEBD['n_layers_II'])
    n_layers_IV = list(TEBD['n_layers_IV'])
    tebd_I = list(TEBD['tebd_I'])
    tebd_II = list(TEBD['tebd_II'])
    tebd_IV = list(TEBD['tebd_IV'])

    # ----- Plot the comparison -----
    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'height_ratios': [2.5, 1]})

    ax1.plot(n_layers_I, tebd_I, 'o-', lw=1, c='darkblue', mec='darkblue', mfc='white', ms=7, label='Trotter I')
    ax1.plot(n_layers_II, tebd_II, 'o-', lw=1, c='darkorange', mec='darkorange', mfc='white', ms=7, label='Trotter II')
    ax1.plot(n_layers_IV, tebd_IV, 'o-', lw=1, c='purple', mec='purple', mfc='white', ms=7, label='Trotter IV')
    ax1.plot(tr_nlayers, tr_test_errors, 'o-', c='darkcyan', ms=7, lw=1, label='Trust-region')
    ax1.set_xlabel('Number of layers')
    ax1.set_ylabel('Empirical test risk')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()

    if hamiltonian == 'ising-1d': id = jnp.linspace(1e-7, 2., 100)
    elif hamiltonian == 'heisenberg': id = jnp.linspace(1e-9, 1e-1, 100)
    min_length = min([len(results['adam_test_errors']), len(results['tr_test_errors'])])
    ax2.plot(id, id, '-', c='k', lw=2, label='Identity')
    ax2.plot(results['tr_test_errors'][:min_length], results['adam_test_errors'][:min_length], 'o', ms=10, mec='white', c='darkcyan')
    ax2.set_ylabel('ADAM')
    ax2.set_xlabel('Trust-region')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True)

    plt.legend()
    fig.tight_layout()

    fname = os.path.join(current_dir, "gfx", f'{str(n_sites)}_{hamiltonian}_results.')

    plt.savefig(fname+'.pdf')
    plt.show()


def plot_convergence_curve(hamiltonian, n_sites, label):

    f_curves = f'{n_sites}_{hamiltonian}_{label}_loss.h5'
    f_curves = os.path.join(data_dir, f_curves)
    curves = save_model.load_results_h5(f_curves)

    adam_curve_test = list(curves['adam_test_errors'])
    tr_curve_test = list(curves['tr_test_errors'])
    tr_hvp_evals = list(curves['tr_hvp_evals'])

    # ----- Plot the comparison -----
    plt.figure()
    plt.plot(adam_curve_test, '-', lw=3, c='darkblue', label='ADAM')
    plt.plot(tr_curve_test, '-', lw=3, c='darkorange', label='Trust-region')
    plt.plot(tr_hvp_evals, tr_curve_test, '-', lw=3, c='darkorange', alpha=0.5, label='Trust-region: HVP evaluations')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Empirical test risk')
    plt.tight_layout()

    fname = os.path.join('gfx', f'{str(n_sites)}_{hamiltonian}_{label}_loss.pdf')
    plt.savefig(fname)
    plt.show()


def main():
    # ----- Plot optimized circuits -----
    plot_optimized_circuit('ising-1d', 50)
    plot_optimized_circuit('heisenberg', 40)

    # ----- Plot comparison of convergence curves -----
    plot_convergence_curve('ising-1d', 50, 'II5')
    plot_convergence_curve('heisenberg', 40, 'IV1')
    plot_convergence_curve('heisenberg', 40, 'IV2')


if __name__ == "__main__":
    main()
