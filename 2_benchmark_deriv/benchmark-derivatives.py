'''
Benchmark the execution time of the derivative functions that are called during the optimization:

    grad : compute_riemannian_loss_gradient_hvp
    hvp  : compute_riemannian_hvp_hvp

The functions are set up exactly as in brickwall_opt.py. Each function is called once
before timing, so the compilation time is excluded from the reported times.

The samples are loaded from 1_circuit-optimization/<hamiltonian>/samples.
However, the timings only depend on the maximum bond dimension.

Usage:
    python3 benchmark-derivatives.py configs/config_ising.yml
    python3 benchmark-derivatives.py configs/config_heisenberg.yml
'''
import os
import csv
import importlib.util
from sys import argv
from copy import deepcopy
from yaml import safe_load
from time import time, perf_counter

# This needs to be set before importing jax!
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
# One device per sample: n_samples_train must be <= device count
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from rqcopt_mps import brickwall_circuit, mps, objective_function, riemannian_manifold, tn_helpers, util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), '1_circuit-optimization')

# Reuse the model initialization and sample loading of run-optimization.py
_spec = importlib.util.spec_from_file_location('run_optimization', os.path.join(OPT_DIR, 'run-optimization.py'))
run_optimization = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_optimization)

ORDER = {'I': 1, 'II': 2, 'IV': 4}


def set_up_model(config):
    ''' Set paths and defaults (without creating a results directory). '''
    config['samples_dir'] = config.get(
        'samples_dir', os.path.join(OPT_DIR, config['hamiltonian'], 'samples'))
    config['benchmark_dir'] = os.path.join(SCRIPT_DIR, 'results')
    os.makedirs(config['benchmark_dir'], exist_ok=True)

    config.setdefault('is_TI', False)
    config.setdefault('cost', 'HST')
    config.setdefault('optimizer', 'trust-region')  # Only needed by initialize_model
    config.setdefault('n_reps', 10)
    config.setdefault('seed', 0)
    return config


def get_model_config(config, label):
    ''' Set degree and n_repetitions of the Trotter circuit from a label like II_3. '''
    order, n_rep = label.split('_')
    model_config = deepcopy(config)
    model_config['degree'] = ORDER[order]
    model_config['n_repetitions'] = int(n_rep)
    return model_config


def get_samples(config):
    '''
    Load the training samples as in run-optimization.py.
    Fall back to Haar-random product states if no matching samples file exists.
    '''
    n = config['n_samples_train']
    try:
        config, psis, phis = run_optimization.load_samples(config)
        return psis[:n], phis[:n], 'file'
    except (IndexError, FileNotFoundError):
        print(f"No samples for n_sites={config['n_sites']} in {config['samples_dir']}; "
              f"using Haar-random product states.")

    k1, k2 = jax.random.split(jax.random.PRNGKey(config['seed'] + 1))
    states = []
    for key in (k1, k2):
        # (n_sites, n, 1, 2, 1) -> (n, n_sites, 1, 2, 1) -> pad to (n, n_sites, chi, 2, chi)
        psi = mps.get_haar_random_product_state_stacked(key, config['n_sites'], n).transpose(1, 0, 2, 3, 4)
        states.append(tn_helpers.pad_mps_to_max_bonddim(psi, config['truncation_dim'], keep_axes=True))
    return states[0], states[1], 'random'


def time_execution(run, n_reps):
    ''' Call once (compilation), then time n_reps executions (blocking until results are ready). '''
    jax.block_until_ready(run())
    times = []
    for _ in range(n_reps):
        t0 = perf_counter()
        jax.block_until_ready(run())
        times.append(perf_counter() - t0)
    return np.array(times)


def benchmark_model(config, qubits, Gs, psis, phis):
    # ----- Derivative functions as defined in brickwall_opt.py -----
    layout = brickwall_circuit.get_circuit_layout(qubits)
    args = (config['truncation_dim'], config['rel_tol'], config['abs_tol'], config['is_TI'])
    grad_fn = getattr(objective_function, f"compute_riemannian_loss_gradient_{config['cost']}")
    hvp_fn = getattr(objective_function, f"compute_riemannian_hvp_{config['cost']}")
    gradfunc = lambda x: grad_fn(psis, phis, x, layout, *args)
    hessfunc = lambda x, eta: hvp_fn(psis, phis, x, eta, layout, *args)

    # Random tangent direction for the HVP
    k1, k2 = jax.random.split(jax.random.PRNGKey(config['seed']))
    Zs = jax.random.normal(k1, Gs.shape) + 1j * jax.random.normal(k2, Gs.shape)
    Zs = riemannian_manifold.project_to_tangent_space(Gs, Zs.astype(Gs.dtype))

    return {
        'grad': time_execution(lambda: gradfunc(Gs), config['n_reps']),
        'hvp': time_execution(lambda: hessfunc(Gs, Zs), config['n_reps']),
    }


def print_summary(config, rows):
    print(f"\n##### Summary ({config['hamiltonian']}, median execution time) #####")
    print(f"{'model':6s} {'layers':>6s} | {'grad [s]':>10s} {'hvp [s]':>10s} {'ratio':>6s}")
    for label in config['models']:
        g = next(r for r in rows if r['model'] == label and r['mode'] == 'grad')
        h = next(r for r in rows if r['model'] == label and r['mode'] == 'hvp')
        print(f"{label:6s} {g['n_layers']:6d} | {g['t_exec_median_s']:10.4f} "
              f"{h['t_exec_median_s']:10.4f} {h['t_exec_median_s']/g['t_exec_median_s']:6.2f}")


def main():
    t0 = time()

    # ----- Load the config file and set up benchmark -----
    with open(argv[1], 'r') as f:
        config = safe_load(f)
    config = set_up_model(config)
    n_devices = jax.local_device_count()
    filename = os.path.join(
        config['benchmark_dir'], f"{config['n_sites']}_{config['hamiltonian']}_derivative_timings.csv")

    print(f'\n##### Derivative benchmark #####\n')
    print(f"JAX {jax.__version__}, backend = {jax.default_backend()}, n_devices = {n_devices}")
    print(f"\t*{config['hamiltonian']} Hamiltonian")
    print(f"\t*n_sites = {config['n_sites']}")
    print(f"\t*cost = {config['cost']}")
    print(f"\t*models = {config['models']}")
    print(f"\t*n_reps = {config['n_reps']}")
    print(f"\t*Optimize TI brickwall circuit = {config['is_TI']}")
    assert config['n_samples_train'] <= n_devices, \
        f"n_samples_train={config['n_samples_train']} > n_devices={n_devices}"

    fields = ['hamiltonian', 'model', 'degree', 'n_repetitions', 'n_sites', 'n_layers', 'n_gates',
              'n_samples', 'samples', 'truncation_dim', 'is_TI', 'cost', 'mode',
              't_exec_median_s', 't_exec_min_s', 't_exec_mean_s', 't_exec_std_s', 'n_reps', 'n_devices']
    rows = []
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for label in config['models']:
            print(f"\n### Model {label} ###")
            model_config = get_model_config(config, label)

            # ----- Initial Trotter circuit and samples (as in run-optimization.py) -----
            model_config, qubits, Gs_start, *_ = run_optimization.initialize_model(model_config)
            psis, phis, samples_source = get_samples(model_config)
            Gs = util.get_tensors(Gs_start)

            # ----- Benchmark -----
            res = benchmark_model(model_config, qubits, Gs, psis, phis)

            for mode, t in res.items():
                print(f"  {mode:5s} median = {np.median(t):.4f} s  (min = {t.min():.4f} s)")
                row = dict(
                    hamiltonian=config['hamiltonian'], model=label, degree=model_config['degree'],
                    n_repetitions=model_config['n_repetitions'], n_sites=model_config['n_sites'],
                    n_layers=model_config['n_layers'], n_gates=model_config['n_gates'],
                    n_samples=psis.shape[0], samples=samples_source,
                    truncation_dim=model_config['truncation_dim'], is_TI=model_config['is_TI'],
                    cost=model_config['cost'], mode=mode,
                    t_exec_median_s=np.median(t), t_exec_min_s=t.min(),
                    t_exec_mean_s=t.mean(), t_exec_std_s=t.std(), n_reps=len(t), n_devices=n_devices,
                )
                writer.writerow(row)
                rows.append(row)
            f.flush()

    print_summary(config, rows)
    print(f'\nResults written to {filename}')

    print(f'\n##### Benchmark finished! #####')
    run_optimization.get_duration(t0, program='script')


if __name__ == "__main__":
    main()
