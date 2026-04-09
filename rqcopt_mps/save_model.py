import h5py
import os
import glob

import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import orbax.checkpoint as ocp

def _convert_attr(val):
    """Convert HDF5 attributes (numpy scalars → Python scalars)."""
    if isinstance(val, (np.bool_,)):
        return bool(val)
    elif isinstance(val, (np.integer,)):
        return int(val)
    elif isinstance(val, (np.floating,)):
        return float(val)
    elif isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8")  # in case strings were stored as bytes
    else:
        return val
    
def _to_numpy(x):
    """Convert JAX arrays / numpy arrays to numpy; pass through python scalars."""
    if isinstance(x, jax.Array):
        return np.asarray(x)  # device -> host
    if hasattr(jnp, "ndarray") and isinstance(x, jnp.ndarray):
        return np.asarray(x)
    if isinstance(x, np.ndarray):
        return x
    return x

def to_jax(x, keep_python_scalars=True):
    """Recursively convert numpy arrays (and optionally numpy scalars) to jax arrays."""
    # dict
    if isinstance(x, dict):
        return {k: to_jax(v, keep_python_scalars=keep_python_scalars) for k, v in x.items()}

    # list / tuple
    if isinstance(x, list):
        return [to_jax(v, keep_python_scalars=keep_python_scalars) for v in x]
    if isinstance(x, tuple):
        return tuple(to_jax(v, keep_python_scalars=keep_python_scalars) for v in x)

    # numpy array
    if isinstance(x, np.ndarray):
        return jnp.asarray(x)

    # numpy scalar
    if isinstance(x, np.generic):
        if keep_python_scalars:
            return x.item()   # -> python float/int/bool
        return jnp.asarray(x)

    # leave everything else untouched (strings, None, python floats, etc.)
    return x
    

# ----- Samples pairs -----
def get_samples_filename(**kwargs):
    if kwargs['hamiltonian'] == 'ising-1d':
        filename = (
            f"samples_nsites{kwargs['n_sites']}_t{kwargs['t']}"
            f"_J{kwargs['J']}_g{kwargs['g']}_h{kwargs['h']}"
            )
    elif kwargs['hamiltonian'] == 'heisenberg':
        filename = (
            f"samples_nsites{kwargs['n_sites']}_t{kwargs['t']}"
            )
    elif kwargs['hamiltonian'] == 'nnn-ising':
        filename = (
            f"samples_nsites{kwargs['n_sites']}_t{kwargs['t']}"
            f"_J1{kwargs['J1']}_J2{kwargs['J2']}_g{kwargs['g']}"
            )
    matches = sorted(glob.glob(os.path.join(kwargs["samples_dir"], filename + "*")))
    return matches[-1]

def save_mps_pairs(filename, initial_states, reference_states, config=None):
    """
    Save a list initial psis and reference phis to an HDF5 file.
    
    pairs: list [psis, phis]
    """
    with h5py.File(filename, "w") as f:
        # Save config as file-level attributes
        if config is not None:
            for key, value in config.items():
                f.attrs[key] = value

        # Store MPS pairs
        initial_states = np.asarray(initial_states)
        reference_states = np.asarray(reference_states)
        f.create_dataset("initial_states",  data=initial_states)
        f.create_dataset("reference_states",  data=reference_states)

def load_reference_data(filename):
    """
    Load MPS pairs and optional initial states from an HDF5 file.
    
    Returns:
        pairs: list of tuples (mps1, mps2)
        params: dict of saved parameters
        initial_states: list of MPS objects or None if not present
    """
    reference_states = []
    params = {}
    initial_states = []

    with h5py.File(filename, "r") as f:
        # Load file-level attributes as Python scalars
        params = {key: _convert_attr(val) for key, val in f.attrs.items()}

        initial_states = jnp.array(f["initial_states"])
        reference_states = jnp.array(f["reference_states"])

    return initial_states, reference_states, params


# ----- Model number -----

def remove_blank_lines(fdir):
    result = ""
    with open(fdir, "r") as file:
        result += "".join(line for line in file if not line.isspace())
        result = os.linesep.join([s for s in result.splitlines() if s])
    with open(fdir, "w") as file:
        file.seek(0)
        file.write(result)
            
def get_model_nbr(model_list):
    with open(model_list, "r") as file:
        for last_line in file:
            model_nbr=int(last_line)+1
        file.close()
        print(model_nbr)
    with open(model_list, "a") as file:
        file.write('\n'+str(model_nbr))
        file.close()
    return model_nbr


# ----- Configuration -----
def save_config(config, status='before_training'):
    # status either 'before_training' or 'after_training' or 'intermediate'

    # Store intermediate config
    if status=='intermediate':
        filename = f"{config['model_nbr']}_config.npy"
        fdir = os.path.join(config['model_dir'], filename)
        _ = jnp.save(fdir, config)

    elif status in ['before_training', 'after_training']:
        filename = f"{config['model_nbr']}_config_{status}.txt"
        fdir = os.path.join(config['model_dir'], filename)
        with open(fdir, 'w') as f:
            if status=='before_training':
                f.write('# Configuration of model before training\n\n')
            elif status=='after_training':
                f.write('# Configuration of model after training\n\n')
            
            for key in config.keys():
                f.write('{}: {}\n'.format(key, config[key]))
                
        if status=='before_training':
            print(f'\n... Configuration before training saved to:\n{fdir}\n')
        if status=='after_training':
            filename = f"{config['model_nbr']}_config.npy"
            fdir = os.path.join(config['model_dir'], filename)
            _ = jnp.save(fdir, config)
            print('\n... Configuration after training saved to:\n{}\n'.format(fdir))

def load_config(config, loaded_config):
    '''
    Load the configuration of an optimized model.

    config: configuration of current model
    model_dir: directory of model to be loaded
    load_which: model number of model to be loaded

    Returns:
    --------
    config: configuration of the current model updated by the configuration of the loaded model
    '''
    
    not_load_keys = [
        'model_nbr', 'model_dir', 'load', 'load_which', 'load_dir', 
        'samples_dir', 'output', 'n_iter', 'radius_init', 'sample_filename'
        ]
    # Overwrite the loaded config
    for key in [
        'degree', 'n_repetitions', 'load_server',
        'truncation_dim', 'n_iter', 'optimizer'
        ]:
        if key in config.keys(): not_load_keys.append(key)
    
    for key in loaded_config.keys():
        if key not in not_load_keys:
            config[key] = loaded_config.get(key)

    return config


# ----- Optimization results -----

def _write_item(h5group, key, value, compression="gzip"):
    """Write a single item to an h5 group."""
    value = _to_numpy(value)

    # Nested dict -> subgroup
    if isinstance(value, dict):
        sub = h5group.require_group(key)
        for k, v in value.items():
            _write_item(sub, k, v, compression=compression)
        return

    # Strings
    if isinstance(value, str):
        dt = h5py.string_dtype(encoding="utf-8")
        h5group.create_dataset(key, data=value, dtype=dt)
        return

    # Bytes
    if isinstance(value, (bytes, bytearray)):
        h5group.create_dataset(key, data=np.void(value))
        return

    # None -> store marker attribute (HDF5 has no None)
    if value is None:
        h5group.attrs[f"{key}__is_none"] = True
        return

    # Arrays
    if isinstance(value, np.ndarray):
        # Use compression for non-scalar arrays
        if value.shape != ():
            h5group.create_dataset(key, data=value, compression=compression)
        else:
            h5group.create_dataset(key, data=value)  # scalar array
        return

    # Python scalars (float/int/bool, numpy scalar types)
    if isinstance(value, (int, float, bool, np.number)):
        h5group.create_dataset(key, data=value)
        return

    # Fallback: store repr
    dt = h5py.string_dtype(encoding="utf-8")
    h5group.create_dataset(key, data=repr(value), dtype=dt)
    h5group.attrs[f"{key}__stored_as_repr"] = True

def save_results_h5(path, results: dict, compression="gzip"):
    """Save a (possibly nested) results dict to an HDF5 file."""
    with h5py.File(path, "w") as f:
        for k, v in results.items():
            _write_item(f, k, v, compression=compression)

def _read_item(h5obj):
    """Read back data; returns numpy arrays / python scalars / dicts."""
    if isinstance(h5obj, h5py.Group):
        out = {}
        for k, v in h5obj.items():
            out[k] = _read_item(v)
        # Recover None markers if present
        for attr_k, attr_v in h5obj.attrs.items():
            if attr_k.endswith("__is_none") and attr_v:
                out[attr_k[:-9]] = None
        return out

    # Dataset
    data = h5obj[()]
    # Decode strings
    if isinstance(data, (bytes, np.bytes_)):
        try:
            return data.decode("utf-8")
        except Exception:
            return data
    return data

def load_results_h5(path):
    with h5py.File(path, "r") as f:
        results = _read_item(f)
    results_jax = to_jax(results, keep_python_scalars=True)
    return results_jax


# ----- Checkpoint -----

def make_ckpt_manager(ckpt_dir: str):
    os.makedirs(ckpt_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=10,
        create=True,
        cleanup_tmp_directories=True,
        save_interval_steps=10**18,
    )
    return ocp.CheckpointManager(ckpt_dir, options=options)

def save_ckpt_trust_region(manager, step, x, f_iter, f_test, radius, n_deriv_evals):
    if jax.process_index() != 0:
        return
    state = {"step": step, "x": x, "f_iter": f_iter, "f_test": f_test, "radius": radius, "n_deriv_evals": n_deriv_evals}
    manager.save(step, args=ocp.args.StandardSave(state), force=True)

def save_ckpt_adam(manager, step, x, f_iter, f_test, m, v):
    if jax.process_index() != 0:
        return
    state = {"step": step, "x": x, "f_iter": f_iter, "f_test": f_test, "m": m, "v": v}
    manager.save(step, args=ocp.args.StandardSave(state), force=True)

def load_specific_checkpoint_trust_region(
    ckpt_dir,
    target_step,
):
    mgr = make_ckpt_manager(ckpt_dir)
    try:
        # Read available steps on disk
        steps = mgr.all_steps(read=True)
        
        # Check if the requested checkpoint actually exists
        if target_step not in steps:
            raise ValueError(
                f"Requested checkpoint step {target_step} not found. "
                f"Available steps in '{ckpt_dir}': {steps}"
            )

        # Attempt to restore the specific step
        state = mgr.restore(target_step) 
        loaded_step = int(state.get("step", target_step))

        if jax.process_index() == 0:
            print(f"[ckpt] Restored checkpoint dir step={target_step}, state['step']={loaded_step}")

        return (
            state["x"],
            state["f_iter"],
            state["f_test"],
            state["radius"],
            state["n_deriv_evals"],
            loaded_step,
        )
        
    except Exception as e:
        # Wrap any restoration errors (e.g. corrupted files) with a clear message
        raise RuntimeError(
            f"Failed to restore specific checkpoint step={target_step}. "
            f"Error: {e}"
        ) from e
        
    finally:
        mgr.close()

def load_latest_checkpoint_trust_region(
    ckpt_dir,
    x_template,
    r_template=1.0,
    i_template=0.
):
    mgr = make_ckpt_manager(ckpt_dir)
    try:
        # Ensure we look at what’s actually on disk (helpful on shared FS)
        steps = sorted(mgr.all_steps(read=True))
        if not steps:
            return x_template, None, None, r_template, i_template, 0

        last_err = None
        for step in reversed(steps):  # try latest, then older
            try:
                state = mgr.restore(step)  # no template => no shape mismatch for variable-length histories
                loaded_step = int(state.get("step", step))

                if jax.process_index() == 0:
                    # `step` is the directory step; `loaded_step` is whatever you stored in state["step"]
                    print(f"[ckpt] Restored checkpoint dir step={step}, state['step']={loaded_step}")

                return (
                    state["x"],
                    state["f_iter"],
                    state["f_test"],
                    state["radius"],
                    state["n_deriv_evals"],
                    loaded_step,
                )
            except Exception as e:
                last_err = e
                continue

        # If we get here, none of the checkpoints could be restored
        raise RuntimeError(
            f"Failed to restore any checkpoint from steps={steps}. Last error: {last_err}"
        ) from last_err
    finally:
        mgr.close()

def load_latest_checkpoint_adam(
    ckpt_dir,
    x_template,
    m_template,
    v_template,
):
    mgr = make_ckpt_manager(ckpt_dir)
    try:
        # Ensure we look at what’s actually on disk (helpful on shared FS)
        steps = sorted(mgr.all_steps(read=True))
        if not steps:
            return x_template, None, None, m_template, v_template, 0

        last_err = None
        for step in reversed(steps):  # try latest, then older
            try:
                state = mgr.restore(step, args=ocp.args.StandardRestore())
                loaded_step = int(state.get("step", step))

                if jax.process_index() == 0:
                    print(f"[ckpt] Restored checkpoint dir step={step}, state['step']={loaded_step}")

                return (
                    state["x"],
                    state.get("f_iter", None),
                    state.get("f_test", None),
                    state["m"],
                    state["v"],
                    loaded_step,
                )
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(
            f"Failed to restore any checkpoint from steps={steps}. Last error: {last_err}"
        ) from last_err
    finally:
        mgr.close()


# ----- end checkpoints -----