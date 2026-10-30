"""
Check the Trotter error scaling of (not optimized) Trotter circuits.

The initial and reference states are loaded from
1_circuit-optimization/<hamiltonian>/samples/. Each Trotter circuit is applied
to the initial MPS and the result is compared to the reference MPS.
The error per sample is sqrt(1 - |<phi_ref|phi_trotter>|^2).

For a Trotter formula of order p, the error at fixed total time t scales as
dt^p with dt = t/n_repetitions. The observed order can be extracted by fitting 
the error in a log-log plot.

Usage: python3 run-trotter-comparison.py configs/config-ising.yml
"""
import os
from sys import argv

from yaml import safe_load

# This needs to be set before importing jax!
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from rqcopt_mps import (
    brickwall_circuit,
    mps,
    save_model,
    spin_systems,
    tebd,
    tn_helpers,
    util,
)

jax.config.update("jax_enable_x64", True)

script_path = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.dirname(script_path)


# |<bra|ket>|^2 per sample, where bra already holds the complex conjugate
batched_fidelity = jax.jit(jax.vmap(mps.compute_fidelity))


def get_samples_path(config):
    samples_dir = os.path.join(repo_path, '1_circuit-optimization', config['hamiltonian'], 'samples')
    return save_model.get_samples_filename(samples_dir=samples_dir, **config)


def load_samples(config):
    ''' Load initial and reference MPS from 1_circuit-optimization/<hamiltonian>/samples/. '''
    filename = get_samples_path(config)
    print("Load samples from", filename)
    initial_states, reference_states, config_samples = save_model.load_reference_data(filename)

    # The samples must belong to the same model as the config
    keys = ['n_sites', 't', 'J', 'h'] + (['g'] if config['hamiltonian'] == 'ising-1d' else [])
    for k in keys:
        assert np.allclose(config[k], config_samples[k]), \
            f"{k}={config[k]} in config but {k}={config_samples[k]} in samples"

    n_samples = config['n_samples']
    assert n_samples <= initial_states.shape[0], \
        f"n_samples={n_samples} but only {initial_states.shape[0]} samples in file"
    initial_states = initial_states[:n_samples]
    reference_states = reference_states[:n_samples].conj()  # Our convention for the bra
    print("Loaded samples", initial_states.shape, reference_states.shape)
    print(f"Reference: degree={config_samples['degree']}, n_repetitions={config_samples['n_repetitions']}")

    # Pad to the chosen maximum bond dimension
    initial_states = tn_helpers.pad_mps_to_max_bonddim(initial_states, config['truncation_dim'], keep_axes=True)
    reference_states = tn_helpers.pad_mps_to_max_bonddim(reference_states, config['truncation_dim'], keep_axes=True)
    return initial_states, reference_states, config_samples


def get_couplings(config):
    ''' Coupling parameters passed to the Trotter gate construction. '''
    if config['hamiltonian'] == 'ising-1d':
        if config['is_TI']:
            return {'J': config['J'], 'g': config['g'], 'h': config['h']}
        _, J, g, h = spin_systems.construct_ising_hamiltonian(
            config['n_sites'], config['J'], config['g'], config['h'],
            disordered=False, get_matrix=False, key=None)
        return {'J': J, 'g': g, 'h': h}
    elif config['hamiltonian'] == 'heisenberg':
        if config['is_TI']:
            return {'J': config['J'], 'h': config['h']}
        _, J, h = spin_systems.construct_heisenberg_hamiltonian(
            config['n_sites'], config['J'], config['h'],
            disordered=False, get_matrix=False, key=None)
        return {'J': J, 'h': h}
    raise ValueError(f"Hamiltonian {config['hamiltonian']} not supported")


def apply_trotter_circuit(psi, t, degree, n_repetitions, couplings, config):
    ''' Apply the Trotter circuit to all initial MPS. Returns the evolved MPS and the number of layers. '''
    circuit, circuit_qubits = tebd.create_tebd_circuit(
        t, config['n_sites'], int(n_repetitions), degree,
        config['hamiltonian'], config['is_TI'], **couplings)
    Gs = jnp.asarray(util.flatten(circuit))
    layout = brickwall_circuit.get_circuit_layout(util.flatten(circuit_qubits))

    phi = tebd.run_tebd(
        psi, Gs, layout.qubits, layout.flat_idxs_fw, layout.layer_ends_fw,
        layout.first_layer_increasing, config['truncation_dim'],
        config['rel_tol'], config['abs_tol'], is_TI=config['is_TI'])
    return phi, len(circuit)


def get_data_path(config, degree):
    return os.path.join(script_path, "data",
                        f"{config['n_sites']}_{config['hamiltonian']}_t{config['t']}_TEBD_{degree}.h5")


def get_run_settings(config):
    ''' Settings the stored Trotter errors depend on (besides degree and n_repetitions). '''
    return {'samples_file': os.path.basename(get_samples_path(config)),
            'n_samples': int(config['n_samples']), 'truncation_dim': int(config['truncation_dim']),
            'rel_tol': float(config['rel_tol']), 'abs_tol': float(config['abs_tol']),
            'is_TI': bool(config['is_TI'])}


def load_trotter_data(path, settings):
    '''
    Load stored Trotter errors as {n_repetitions: (errors per sample, n_layers)}.
    Returns an empty dict if there is no file or it was computed with other settings.
    '''
    if not os.path.exists(path):
        return {}
    data = save_model.load_results_h5(path)
    stored = data['settings']
    for k, v in settings.items():
        same = str(stored[k]) == v if isinstance(v, str) else np.isclose(float(stored[k]), float(v))
        if not same:
            print(f"Stored data in {path} has {k}={stored[k]} instead of {v}, recompute")
            return {}
    return {int(n): (np.asarray(err), int(nl))
            for n, err, nl in zip(data['n_repetitions'], data['tebd_errors'], data['n_layers'])}


def save_trotter_data(path, settings, data):
    ns = sorted(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_model.save_results_h5(path, {
        'settings': settings,
        'n_repetitions': np.asarray(ns),
        'tebd_errors': np.asarray([data[n][0] for n in ns]),  # (n_points, n_samples)
        'n_layers': np.asarray([data[n][1] for n in ns]),
        })


def fit_slope(dts, errs):
    ''' Fit err = a * dt^slope in log-log space. '''
    slope, log_a = np.polyfit(np.log(dts), np.log(errs), 1)
    return slope, np.exp(log_a)


def main():
    with open(argv[1], 'r') as f:
        config = safe_load(f)
    config.setdefault('is_TI', True)
    config.setdefault('rel_tol', 1e-11)
    config.setdefault('abs_tol', 1e-14)
    config.setdefault('recompute', False)  # True ignores stored Trotter data
    config.setdefault('store_data', False)  # True saves newly computed Trotter data to data/

    t = config['t']
    couplings = get_couplings(config)
    settings = get_run_settings(config)
    psi, phi_ref = None, None  # Only loaded if something has to be computed

    results = []
    for i, degree in enumerate(config['degree']):
        n_repetitions = np.arange(config['n_repetitions_start'][i], config['n_repetitions_end'][i]+1)
        dts = t/n_repetitions

        # Reuse stored errors and only compute the missing n_repetitions
        path = get_data_path(config, degree)
        data = {} if config['recompute'] else load_trotter_data(path, settings)
        missing = [int(n) for n in n_repetitions if int(n) not in data]
        print(f"degree {degree}: {len(n_repetitions)-len(missing)} points loaded from {path}, "
              f"{len(missing)} to compute")
        for n in missing:
            if psi is None:
                psi, phi_ref, _ = load_samples(config)
            print(f"degree {degree}, n_repetitions {n}", flush=True)
            phi, nl = apply_trotter_circuit(psi, t, degree, n, couplings, config)
            fid = batched_fidelity(phi_ref, phi)
            data[n] = (np.sqrt(np.abs(1. - np.asarray(fid))), nl)
        if missing and config['store_data']:
            save_trotter_data(path, settings, data)

        errors = np.asarray([data[int(n)][0] for n in n_repetitions])  # (n_points, n_samples)
        n_layers = [data[int(n)][1] for n in n_repetitions]
        mean_error = np.sqrt(np.abs(1. - np.mean(1. - errors**2, axis=1)))  # sqrt(1 - mean fidelity)

        # Fit the slope, skipping the first points (large time steps)
        mask = np.zeros(len(dts), dtype=bool)
        mask[config['n_fit_skip'][i]:] = True
        if mask.sum() >= 2:
            slope, a = fit_slope(dts[mask], mean_error[mask])
        else:
            slope, a = np.nan, np.nan
        print(f"degree {degree}: fitted order {slope:.2f}")

        results.append({'degree': degree, 'dts': dts, 'n_repetitions': n_repetitions,
                        'errors': errors, 'mean_error': mean_error, 'n_layers': np.asarray(n_layers),
                        'fit_mask': mask, 'slope': slope, 'a': a})

    # ----- Plot errors and fitted slopes -----
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for res, c in zip(results, colors):
        dts = res['dts']
        # Single samples (faint) and sample mean
        ax.plot(dts, res['errors'], 'o', color=c, alpha=0.15, ms=3)
        ax.plot(dts, res['mean_error'], 'o-', color=c, label=f"degree {res['degree']}")
        if not np.isnan(res['slope']):
            dt_fit = dts[res['fit_mask']]
            ax.plot(dt_fit, res['a']*dt_fit**res['slope'], 'k--', lw=1,
                    label=rf"fit $\sim \Delta t^{{{res['slope']:.2f}}}$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Trotter step $\Delta t = t/n$')
    ax.set_ylabel(r'$\sqrt{1 - |\langle \phi_\mathrm{ref} | \phi_\mathrm{Trotter}\rangle|^2}$')
    ax.set_title(f"{config['hamiltonian']}, {config['n_sites']} sites, t={t}, {config['n_samples']} samples")
    ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.join(script_path, "gfx"), exist_ok=True)
    fname = os.path.join(script_path, "gfx", f"{config['n_sites']}_{config['hamiltonian']}_trotter.pdf")
    fig.savefig(fname)
    print("Saved plot to", fname)
    plt.show()


if __name__ == "__main__":
    main()
