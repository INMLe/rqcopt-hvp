import glob
import os
import sys

import yaml

# This needs to be set before importing jax!
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=16"

import jax.numpy as jnp
from jax import config, random

from rqcopt_mps import (
    brickwall_circuit,
    mps,
    save_model,
    spin_systems,
    tebd,
    tn_helpers,
    util,
)

config.update("jax_enable_x64", True)


def is_sample_present(sample, history):
    element_match = (history == sample)
    full_sample_match = jnp.all(element_match, axis=(1, 2, 3, 4))
    return jnp.any(full_sample_match)


def compute_compressed_mps(phi, bond_dim, err_threshold, step_size, rel_tol, abs_tol):
    ''' Compress an MPS before padding. '''

    print("Start compression ...")
    bond_dim_comp = bond_dim
    err1 = 0.
    phi_unpad = tn_helpers.unpad_mps_to_list(phi)
    phi_comp = phi_unpad.copy()
    while err1 < err_threshold:
        err1_prev = err1
        phi_prev = phi_comp.copy()
        phi_comp = phi_unpad.copy()

        bond_dim_comp -= step_size
        phi_comp = mps.compress_mps(phi_comp, bond_dim_comp, rel_tol, abs_tol)
        phi_comp_pad = tn_helpers.pad_mps_to_max_bonddim(phi_comp, bond_dim)
        err1 = 1 - mps.compute_fidelity(phi_comp_pad.conj(), phi)
        print(f"\t Errors for bond dims {bond_dim_comp}: {err1}")

    # Get information about final reference
    phi = phi_prev
    compressed_bondim = bond_dim_comp+step_size
    print('Final MPS has maximum bond dimension: ', compressed_bondim)
    print('Final error between reference MPS and converged MPS: ', err1_prev)

    # Pad to new maximum bond dimension
    phi_pad = tn_helpers.pad_mps_to_max_bonddim(phi, compressed_bondim)

    return phi_pad, err1_prev, compressed_bondim


def main():
    print("*****"*3, "Run TEBD", "*****"*3)

    # Read in config
    with open(sys.argv[1], 'r') as f:
        config = yaml.safe_load(f)

    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_dir = os.path.join(script_dir, config['hamiltonian'], 'samples')

    # Random keys
    base_key = random.PRNGKey(config["reference_seed"])
    key, next_key = random.split(base_key, num=2)

    # Generate Hamiltonian
    if config['hamiltonian'] == 'ising-1d':
        # Set filename for storing
        filename = ('samples_nsites' + str(config['n_sites']) + '_t' + str(config['t']) +
                "_J"+str(config["J"]) + "_g"+str(config["g"]) + "_h"+str(config["h"]))
        filename2 = ('samples_nsites' + str(config['n_sites']) + '_t' + str(config['t']) +
                "_J"+str(config["J"]) + "_g"+str(config["g"]) + "_h"+str(config["h"]))

        # Construct the Hamiltonian
        if not config['is_TI']:
            _, Js, gs, hs = spin_systems.construct_ising_hamiltonian(
                config['n_sites'], config['J'], config['g'], config['h'],
                config['disordered'], get_matrix=False, key=key
                )
        else:
            Js, gs, hs = config['J'], config['g'], config['h']
        kwargs = {"J": Js, "g": gs, "h": hs}

    elif config['hamiltonian'] == 'heisenberg':
        # Set filename for storing
        filename = ('samples_nsites' + str(config['n_sites']) + '_t' + str(config['t']) +
                    "_J"+str(config["J"]) + "_h"+str(config["h"]))
        filename2 = ('samples_nsites' + str(config['n_sites']) + '_t' + str(config['t']))

        # Construct the Hamiltonian
        if not config['is_TI']:
            _, Js, hs = spin_systems.construct_heisenberg_hamiltonian(
                config['n_sites'], config['J'], config['h'],
                config['disordered'], get_matrix=False, key=key
                )
        else:
            Js, hs = config['J'], config['h']
        kwargs = {"J": Js, "h": hs}

    # Create TEBD circuit
    circuit, qubits = tebd.create_tebd_circuit(
        config['t'], config['n_sites'], config['n_repetitions'],
        config['degree'], hamiltonian=config['hamiltonian'],
        is_TI=config['is_TI'], **kwargs
        )
    Gs = jnp.asarray(util.flatten(circuit))
    qubits = util.flatten(qubits)

    if config['load_samples']:
        matches = sorted(glob.glob(os.path.join(sample_dir, filename2 + "*")))
        if not matches:
            raise FileNotFoundError(f"No files found starting with '{filename2}' in {sample_dir}")
        filename = matches[-1]
        path = os.path.join(sample_dir, filename)

        initial_states_loaded, reference_states_loaded, config_loaded = save_model.load_reference_data(path)
        print(f"... {len(initial_states_loaded)} samples loaded!")
        print(f"... loaded reference states of shape {reference_states_loaded.shape}")
        key, next_key = random.split(config_loaded['rnd_key'], num=2)

        # Update filename for storing
        if config['hamiltonian'] == 'ising-1d':
            filename = ('samples_nsites' + str(config['n_sites']) + '_t' + str(config['t']) +
                    "_J"+str(config["J"]) + "_g"+str(config["g"]) + "_h"+str(config["h"]))
        elif config['hamiltonian'] == 'heisenberg':
            filename = ('samples_nsites' + str(config['n_sites']) + '_t' + str(config['t']) +
                    "_J"+str(config["J"]) + "_h"+str(config["h"]))
        filename += '_nsamples' + str(config['n_samples']+len(initial_states_loaded)) + '.h5'
        path = os.path.join(sample_dir, filename)

    else:
        filename += '_nsamples' + str(config['n_samples']) + '.h5'
        path = os.path.join(sample_dir, filename)

    if config["compress"]:
        step_size = 1
        err_threshold = float(config["err_thres"])
        print("Error threshold = ", err_threshold)
        err_phi_max = 0.
    else:
        err_phi_max = None

    # Generate batched Haar random MPS
    psi = mps.get_haar_random_product_state_stacked(next_key, config['n_sites'], config['n_samples'], d=2)
    print("Original Haar random product state has shape", psi.shape)
    psi = tn_helpers.pad_mps_to_max_bonddim(psi, config["truncation_dim"])  # Pad to zero
    print("Padded Haar random product state has shape", psi.shape)
    config['rnd_key'] = next_key

    # Gate order of the TEBD circuit
    layout = brickwall_circuit.get_circuit_layout(qubits)

    # Evolve initial states by TEBD circuit
    phi = tebd.run_tebd(
        psi, Gs, layout.qubits, layout.flat_idxs_fw, layout.layer_ends_fw,
        layout.first_layer_increasing, config['truncation_dim'],
        config['rel_tol'], config['abs_tol'], config['is_TI']
        )

    if config["compress"]:
        ''' Compression needs to be performed on each MPS separately. '''
        phi_compressed = []
        if config['load_samples']: compressed_bond_dim = reference_states_loaded.shape[2]
        else: compressed_bond_dim = 0
        bond_dim = phi.shape[2]
        for phi_sample in phi:
            phi_sample, err_phi, compressed_bond_dim_ = compute_compressed_mps(
                phi_sample, bond_dim, err_threshold,
                step_size, config['rel_tol'], config['abs_tol']
                )
            phi_compressed.append(phi_sample)
            compressed_bond_dim = max(compressed_bond_dim, compressed_bond_dim_)
            print("compressed bond idm", compressed_bond_dim)
            if err_phi > err_phi_max: err_phi_max = err_phi

        for i in range(len(phi_compressed)):
            phi_sample = phi_compressed[i]
            if phi_sample.shape[1] < compressed_bond_dim:
                phi_compressed[i] = tn_helpers.pad_mps_to_max_bonddim(phi_sample, compressed_bond_dim)

        reference_states = jnp.asarray(phi_compressed)
        config["err_compression"] = err_phi_max

        print("\nThe compressed reference MPS have a maximum bond dimension of", compressed_bond_dim)
        print("The maximum compression error is", err_phi_max)
        config['truncation_dim_eff'] = compressed_bond_dim

    else:
        reference_states = phi
        config['truncation_dim_eff'] = config['truncation_dim']

    psi = psi[:, :, :1, :, :1]  # Get the minimal bond dimension for product states

    if config['load_samples']:
        print(reference_states_loaded.shape, compressed_bond_dim)
        if reference_states_loaded.shape[2] < compressed_bond_dim:
            reference_states_loaded = tn_helpers.pad_mps_to_max_bonddim(
                reference_states_loaded, compressed_bond_dim, keep_axes=True)

        found = []
        for new_sample in reference_states:
            found.append(is_sample_present(new_sample, reference_states_loaded))
        print("All samples new: ", not all(jnp.asarray(found)))
        config['n_samples'] = config['n_samples']+len(initial_states_loaded)

        # Combine loaded and newly generated samples
        psi = jnp.concatenate((initial_states_loaded, psi), axis=0)
        reference_states = jnp.concatenate((reference_states_loaded, reference_states), axis=0)

    # Store the reference
    path = os.path.join(sample_dir, filename)
    _ = save_model.save_mps_pairs(path, psi, reference_states, config)

    print("initial states: ", psi.shape)
    print("reference states: ", reference_states.shape)

    print("Samples saved to disk:", path)


if __name__ == "__main__":
    main()
