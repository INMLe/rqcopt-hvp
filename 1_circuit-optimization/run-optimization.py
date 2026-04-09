import os
from sys import argv
from yaml import safe_load
from time import time

# This needs to be set before importing jax!
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from rqcopt_mps import *

def get_duration(t0, program='script'):
    t1 = time()
    duration = t1-t0; unit='s'
    if duration > 60.: duration=duration/60.; unit='min'
    if duration > 60.: duration=duration/60.; unit='h'
    print(f'\nDuration of {program}: {duration} {unit}\n\n')

def set_up_model(config, path):
    # ----- Set script path -----
    config['script_path'] = path
    hamiltonian_path = os.path.join(path, config['hamiltonian'])    

    # ----- Set the current model number -----
    model_list = os.path.join(hamiltonian_path, 'model_list.txt')
    save_model.remove_blank_lines(model_list)
    model_nbr = save_model.get_model_nbr(model_list)
    config['model_nbr'] = model_nbr

    # ----- Set the directory for loading results (in case) -----
    if config['load']:
        config['load_dir'] = os.path.join(hamiltonian_path, 'results', str(config['load_which']))
        
    # Set the directory for saving results
    model_dir = os.path.join(hamiltonian_path, 'results', str(model_nbr))
    config['model_dir'] = model_dir
    if not os.path.isdir(model_dir): os.makedirs(model_dir)
        

    # Set the reference directory
    config['samples_dir'] = os.path.join(hamiltonian_path, 'samples')    

    # Set default values
    if 'is_TI' not in config.keys(): config['is_TI'] = False
    if 'n_id_layers' not in config.keys(): config['n_id_layers'] = 0
    if 'use_tebd' not in config.keys(): config['use_tebd'] = True

    if config['hamiltonian'] in ['ising-1d', 'heisenberg']: 
        config['first_layer_odd'] = True
    
    return config, model_nbr


def run_optimization(
        config, Gs_start, qubits, initial_states, reference_states, 
        err_iter, err_test_iter, optimizer_params=[]
        ):
    
    t_start = time()

    if config['optimizer']=='trust-region':
        Gs_opt, err_iter, err_test_iter, radius, n_deriv_evals = brickwall_opt.optimize_brickwall_circuit_TrustRegion(
            config, initial_states, reference_states, Gs_start, qubits, config['cost'], err_iter, err_test_iter, optimizer_params)
    elif config['optimizer']=='adam':
        Gs_opt, err_iter, err_test_iter, m, v = brickwall_opt.optimize_brickwall_circuit_ADAM(
            config, initial_states, reference_states, Gs_start, qubits, config['cost'], 
            err_iter, err_test_iter, optimizer_params)
        
    get_duration(t_start)

    # ----- Save optimization results -----
    err_iter, err_init, err_opt, _, err_test_iter, err_init_test, err_opt_test, _ = brickwall_opt.postprocessing_loss(
        err_iter, err_test_iter, verbose=False)
    results = {
        'config': config, 'initial_states': initial_states, 'reference_states': reference_states, 'Gs': Gs_opt, 'qubits': qubits,
        'err_iter': err_iter, 'err_init': err_init, 'err_opt': err_opt, 
        'err_test_iter': err_test_iter, 'err_init_test': err_init_test, 'err_opt_test': err_opt_test, 
        }
    
    if config['optimizer'] == 'trust-region':
        results['radius'] = radius
        results['n_deriv_evals'] = n_deriv_evals
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
    if config_loaded['optimizer']=='trust-region':
        Gs_start, err_iter, err_test_iter, config['radius_init'], n_deriv_evals, _ = save_model.load_latest_checkpoint_trust_region(
            config['load_dir'],
            x_template=loaded_model['Gs'],
            r_template=loaded_model['radius'],
            i_template=loaded_model['n_deriv_evals']
            )
        n_deriv_evals = list(n_deriv_evals)
        optimizer_params = [n_deriv_evals]
    elif config_loaded['optimizer']=='adam':
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
    if config['hamiltonian']=='ising-1d':
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
            n_repetitions=int(config['n_repetitions']), first_layer_odd=config['first_layer_odd'],
            J=J, g=g, h=h
            )

    elif config['hamiltonian']=='heisenberg':
        if config['is_TI']:
            J, h = config['J'], config['h']
        else:
            _, J, h = spin_systems.construct_heisenberg_hamiltonian(
                config['n_sites'], config['J'], config['g'], config['h'], 
                config['disordered'], get_matrix=False, key=None
                )
            
        circuit, circuit_qubits = brickwall_circuit.get_initial_gates(
            config['n_sites'], config['is_TI'], 
            t=config['t'], hamiltonian=config['hamiltonian'], degree=config['degree'], 
            n_repetitions=int(config['n_repetitions']), first_layer_odd=config['first_layer_odd'],
            J=J, h=h
            )

    elif config['hamiltonian']=='nnn-ising':
        circuit, circuit_qubits = brickwall_circuit.get_initial_gates(
            config['n_sites'], config['n_id_layers'], config['use_tebd'], config['is_TI'], 
            t=config['t'], hamiltonian=config['hamiltonian'], degree=config['degree'], 
            n_repetitions=int(config['n_repetitions']), first_layer_odd=config['first_layer_odd'],
            J1=config['J1'], J2=config['J2'], g=config['g']
            )
    
    Gs_start = jnp.asarray(util.flatten(circuit))
    qubits = util.flatten(circuit_qubits)
    config['n_layers'] = len(circuit)
    config['n_gates'] = len(Gs_start)
    print(f"\t*n_layers = {config['n_layers']}")
    print(f"\t*n_gates = {config['n_gates']}")


    if config['optimizer']=='adam':
        m = jnp.zeros_like(Gs_start)
        v = jnp.zeros((config['n_gates'],))
        optimizer_params = [m ,v]
    elif config['optimizer']=='trust-region':
        optimizer_params = [[0]]

    return config, qubits, Gs_start, err_iter, err_test_iter, optimizer_params


def load_samples(config):
    path = save_model.get_samples_filename(**config)
    config['sample_filename'] = path
    initial_states, reference_states, config_samples = save_model.load_reference_data(path)
    assert (config['n_samples_train']+config['n_samples_test']) <= initial_states.shape[0]
    assert config['t']==config_samples['t'], f"t={config['t']} for model but t={config_samples['t']} for training samples"
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
    results = {'config': config, 'initial_states': initial_states, 
               'reference_states': reference_states, 
               'Gs': Gs_start, 'qubits': qubits,
               'err_iter': jnp.asarray(err_iter), 'err_init': 1., 
               'err_opt': 1., 'err_test_iter': jnp.asarray(err_test_iter), 
               'err_init_test': 1., 'err_opt_test': 1.}
    
    if config['optimizer']=='trust-region': 
        results['radius'] =  config['radius_init']
        results['n_deriv_evals'] = optimizer_params[0]
    elif config['optimizer']=='adam':
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