import os
from sys import argv
from time import time

from yaml import safe_load

# This needs to be set before importing jax!
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

import jax
import jax.numpy as jnp

from rqcopt_mps import (
    brickwall_circuit,
    brickwall_opt,
    save_model,
    spin_systems,
    tn_helpers,
    util,
)

jax.config.update("jax_enable_x64", True)


def get_duration(t0, program='script'):
    t1 = time()
    duration = t1-t0; unit = 's'
    if duration > 60.: duration = duration/60.; unit = 'min'
    if duration > 60.: duration = duration/60.; unit = 'h'
    print(f'\nDuration of {program}: {duration} {unit}\n\n')


def get_model_nbr(results_dir):
    """
    Return the next free model number: one more than the largest numbered
    folder in results_dir, or 1 if there is none yet.
    The model folder is created right away, so runs started at the same time
    never get the same number.
    """
    os.makedirs(results_dir, exist_ok=True)
    existing = [
        int(d) for d in os.listdir(results_dir)
        if d.isdigit() and os.path.isdir(os.path.join(results_dir, d))
        ]
    model_nbr = max(existing, default=0) + 1
    while True:
        try:
            os.makedirs(os.path.join(results_dir, str(model_nbr)))
            return model_nbr
        except FileExistsError:
            model_nbr += 1


def set_up_model(config, path):
    # ----- Set script path -----
    config['script_path'] = path
    hamiltonian_path = os.path.join(path, config['hamiltonian'])

    # ----- Set the current model number (also creates the model directory) -----
    results_dir = os.path.join(hamiltonian_path, 'results')
    model_nbr = get_model_nbr(results_dir)
    config['model_nbr'] = model_nbr

    # ----- Set the directory for loading results (in case) -----
    if config['load']:
        config['load_dir'] = os.path.join(hamiltonian_path, 'results', str(config['load_which']))

    # Set the directory for saving results
    config['model_dir'] = os.path.join(results_dir, str(model_nbr))

    # Set the reference directory
    config['samples_dir'] = os.path.join(hamiltonian_path, 'samples')

    # Set default values
    if 'is_TI' not in config.keys(): config['is_TI'] = False
    # Number of most recent checkpoints to keep (null in the YAML keeps all)
    if 'n_ckpt_keep' not in config.keys(): config['n_ckpt_keep'] = 10

    return config, model_nbr


def run_optimization(
        config, Gs_start, qubits, initial_states, reference_states,
        err_iter, err_test_iter, optimizer_params=[]
        ):

    t_start = time()

    if config['optimizer'] == 'trust-region':
        Gs_opt, err_iter, err_test_iter, radius, n_deriv_evals = brickwall_opt.optimize_brickwall_circuit_TrustRegion(
            config, initial_states, reference_states, Gs_start, qubits, config['cost'], err_iter, err_test_iter, optimizer_params)
    elif config['optimizer'] == 'adam':
        Gs_opt, err_iter, err_test_iter, m, v = brickwall_opt.optimize_brickwall_circuit_ADAM(
            config, initial_states, reference_states, Gs_start, qubits, config['cost'],
            err_iter, err_test_iter, optimizer_params)

    get_duration(t_start)

    # ----- Save optimization results -----
    err_iter, err_init, err_opt, _, err_test_iter, err_init_test, err_opt_test, _ = brickwall_opt.postprocessing_loss(
        err_iter, err_test_iter, verbose=False)
    results = {
        'config': config, 'Gs': Gs_opt, 'qubits': qubits,
        'err_iter': err_iter, 'err_init': err_init, 'err_opt': err_opt,
        'err_test_iter': err_test_iter, 'err_init_test': err_init_test, 'err_opt_test': err_opt_test,
        }

    if config['optimizer'] == 'trust-region':
        results['radius'] = radius
        results['n_deriv_evals'] = jnp.asarray(n_deriv_evals)  # Array, so it is not stored as text
    elif config['optimizer'] == 'adam':
        results['m'] = m
        results['v'] = v

    filename = os.path.join(config['model_dir'], str(config['model_nbr'])+"_opt.h5")
    _ = save_model.save_results_h5(filename, results)

    return Gs_opt, results


def load_preoptimized_model(config):

    # ---- Load preoptimized model -----
    path = os.path.join(config['load_dir'], str(config['load_which'])+"_opt.h5")
    print("Load preoptimized model from: ", path)
    loaded_model = save_model.load_results_h5(path)
    config_loaded = loaded_model['config']
    config = save_model.load_config(config, config_loaded)
    qubits = loaded_model['qubits']

    # ----- Load from checkpoint -----
    if config_loaded['optimizer'] == 'trust-region':
        Gs_start, err_iter, err_test_iter, config['radius_init'], n_deriv_evals, _ = save_model.load_latest_checkpoint_trust_region(
            config['load_dir'],
            x_template=loaded_model['Gs'],
            r_template=loaded_model['radius'],
            i_template=loaded_model['n_deriv_evals']
            )
        n_deriv_evals = list(n_deriv_evals)
        optimizer_params = [n_deriv_evals]
    elif config_loaded['optimizer'] == 'adam':
        Gs_start, err_iter, err_test_iter, m, v, _ = save_model.load_latest_checkpoint_adam(
            config['load_dir'],
            x_template=loaded_model['Gs'],
            m_template=loaded_model['m'],
            v_template=loaded_model['v'],
            )
        optimizer_params = [m, v]

    err_iter = list(err_iter)
    err_test_iter = list(err_test_iter)

    return config, qubits, Gs_start, err_iter, err_test_iter, optimizer_params


def initialize_model(config):

    err_iter, err_test_iter = [], []

    # ----- Create the Trotter circuit -----
    print("Get the initial Trotter gates ...")
    if config['hamiltonian'] == 'ising-1d':
        if config['is_TI']:
            J, g, h = config['J'], config['g'], config['h']
        else:
            _, J, g, h = spin_systems.construct_ising_hamiltonian(
                config['n_sites'], config['J'], config['g'], config['h'],
                config['disordered'], get_matrix=False, key=None
                )

        circuit, circuit_qubits = brickwall_circuit.get_initial_gates(
            config['n_sites'], config['is_TI'],
            t=config['t'], hamiltonian=config['hamiltonian'], degree=config['degree'],
            n_repetitions=int(config['n_repetitions']),
            J=J, g=g, h=h
            )

    elif config['hamiltonian'] == 'heisenberg':
        if config['is_TI']:
            J, h = config['J'], config['h']
        else:
            _, J, h = spin_systems.construct_heisenberg_hamiltonian(
                config['n_sites'], config['J'], config['h'],
                config['disordered'], get_matrix=False, key=None
                )

        circuit, circuit_qubits = brickwall_circuit.get_initial_gates(
            config['n_sites'], config['is_TI'],
            t=config['t'], hamiltonian=config['hamiltonian'], degree=config['degree'],
            n_repetitions=int(config['n_repetitions']),
            J=J, h=h
            )

    Gs_start = jnp.asarray(util.flatten(circuit))
    qubits = util.flatten(circuit_qubits)
    config['n_layers'] = len(circuit)
    config['n_gates'] = len(Gs_start)
    print(f"\t*n_layers = {config['n_layers']}")
    print(f"\t*n_gates = {config['n_gates']}")

    if config['optimizer'] == 'adam':
        m = jnp.zeros_like(Gs_start)
        v = jnp.zeros((config['n_gates'],))
        optimizer_params = [m, v]
    elif config['optimizer'] == 'trust-region':
        optimizer_params = [[0]]

    return config, qubits, Gs_start, err_iter, err_test_iter, optimizer_params


def load_samples(config):
    # A resumed model reuses the samples file of the loaded model.
    # Otherwise, take the newest matching file in the samples directory.
    if config.get('sample_filename'):
        filename = config['sample_filename']
    else:
        filename = save_model.get_samples_filename(**config)
    # Only store the file name; the file always lives in config['samples_dir']
    filename = os.path.basename(filename)
    config['sample_filename'] = filename
    path = os.path.join(config['samples_dir'], filename)
    initial_states, reference_states, config_samples = save_model.load_reference_data(path)
    assert (config['n_samples_train']+config['n_samples_test']) <= initial_states.shape[0]
    assert config['t'] == config_samples['t'], f"t={config['t']} for model but t={config_samples['t']} for training samples"
    n_samples = config['n_samples_train']+config['n_samples_test']
    initial_states = initial_states[:n_samples]
    reference_states = reference_states[:n_samples]
    reference_states = reference_states.conj()  # Our convention for inner product

    # Preprocessing for loaded data
    print("Loaded samples", initial_states.shape, reference_states.shape)
    initial_states = tn_helpers.pad_mps_to_max_bonddim(initial_states, config['truncation_dim'], keep_axes=True)
    reference_states = tn_helpers.pad_mps_to_max_bonddim(reference_states, config['truncation_dim'], keep_axes=True)

    return config, initial_states, reference_states


def main():
    t0 = time()

    # ----- Load the config file and set up model -----
    with open(argv[1], 'r') as f:
        config = safe_load(f)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config, model_nbr = set_up_model(config, current_dir)

    if config['load']:
        config, qubits, Gs_start, err_iter, err_test_iter, optimizer_params = load_preoptimized_model(config)
    else:
        config, qubits, Gs_start, err_iter, err_test_iter, optimizer_params = initialize_model(config)

    save_model.save_config(config, status='before_training')

    # ----- Print simulation information ------
    print(f'\n##### Simulation for model {model_nbr} #####\n')
    print(f'System with ...')
    print(f"\t*{config['hamiltonian']} Hamiltonian")
    print(f"\t*n_sites = {config['n_sites']}")
    print(f"\t*degree = {config['degree']}")
    print(f"\t*n_repetitions = {config['n_repetitions']}")
    print(f"\t*Optimize TI brickwall circuit = {config['is_TI']}")

    # ----- Load samples -----
    config, initial_states, reference_states = load_samples(config)

    # ----- Store results before optimization -----
    # The samples are not stored. config['sample_filename'] (relative to the samples/ directory),
    # 'n_samples_train', 'n_samples_test' and 'truncation_dim' are enough to reload them with load_samples.
    results = {'config': config,
               'Gs': Gs_start, 'qubits': qubits,
               'err_iter': jnp.asarray(err_iter), 'err_init': 1.,
               'err_opt': 1., 'err_test_iter': jnp.asarray(err_test_iter),
               'err_init_test': 1., 'err_opt_test': 1.}

    if config['optimizer'] == 'trust-region':
        results['radius'] = config['radius_init']
        results['n_deriv_evals'] = jnp.asarray(optimizer_params[0])
    elif config['optimizer'] == 'adam':
        results['m'], results['v'] = optimizer_params

    filename = os.path.join(config['model_dir'], str(config['model_nbr'])+"_opt.h5")
    _ = save_model.save_results_h5(filename, results)

    # ----- Run optimization of model -----
    print(f"\n### Run model for t={config['t']} ###")
    Gs_opt, results = run_optimization(
        config, Gs_start, qubits, initial_states, reference_states,
        err_iter, err_test_iter, optimizer_params
        )

    print(f'\n##### Simulation finished! #####')
    save_model.save_config(config, status='after_training')

    get_duration(t0, program='script')


if __name__ == "__main__":
    main()
