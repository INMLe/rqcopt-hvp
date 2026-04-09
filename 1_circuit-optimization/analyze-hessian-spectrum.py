import os
import argparse
from sys import argv
from yaml import safe_load

nbatch = 8
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={nbatch}"
os.environ["JAX_PLATFORMS"] = "cpu"      # preferred

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

from rqcopt_mps import *


def setup_config(config):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config["load_dir"] = os.path.join(current_dir, config["hamiltonian"], "results", str(config["load_which"]))
    config["samples_dir"] = os.path.join(current_dir, config["hamiltonian"], "samples")
    config["spectrum_dir"] = os.path.join(current_dir, config["hamiltonian"], "spectrum")

    if config["hamiltonian"] == "ising-1d":
        fname = "/samples_nsites8_t2.0_J1.0_g0.75_h0.6_nsamples112.h5"
    elif config["hamiltonian"] == "heisenberg":
        fname = "/samples_nsites8_t0.25_J[1.0, 1.0, -0.5]_h[0.75, 0.0, 0.0]_nsamples112.h5"
    config["sample_filename"] = config["samples_dir"] + fname

    return config



def load_preoptimized_model(config):
    # ---- Load preoptimized model -----
    path = os.path.join(config["load_dir"], str(config["load_which"]) + "_opt.h5")
    loaded_model = save_model.load_results_h5(path)
    old_sample_filename = config["sample_filename"]
    config_loaded = loaded_model["config"]
    config_loaded["sample_filename"] = old_sample_filename
    config = save_model.load_config(config, config_loaded)
    config["n_samples"] = int(config_loaded["n_samples_train"])
    qubits = loaded_model["qubits"]
    return config, qubits


# ── Fixed parameters ──
paulis = {
    "I": np.array(np.eye(2).astype(complex)),
    "x": np.array(np.array([[0, 1], [1, 0]]).astype(complex)),
    "y": np.array(np.array([[0, -1j], [1j, 0]]).astype(complex)),
    "z": np.array(np.array([[1, 0], [0, -1]]).astype(complex)),
}


def get_two_qubit_skew_hermitian_pauli_basis():
    """Return basis {i (P_a ⊗ P_b)} with shape (16, 2, 2, 2, 2)."""
    labels = ("I", "x", "y", "z")
    basis = []
    basis_names = []
    for left in labels:
        for right in labels:
            kron = np.kron(paulis[left], paulis[right])
            basis.append(1j * kron.reshape(2, 2, 2, 2))
            basis_names.append(f"i({left}⊗{right})")
    return jnp.asarray(basis), basis_names


def real_frobenius_inner(a, b):
    """Real Frobenius inner product on complex arrays."""
    return jnp.real(jnp.vdot(a, b))


def get_hessian(qubits, ket, bra, Gs, chi_max, nlayers, project_basis_to_tangent=True):
    """Evaluate HVP on a basis and assemble Hessian coordinates.

    For non-orthonormal tangent bases, self-adjointness should be checked as
    (G H) = (G H)^T, where G is the Gram matrix of the basis.
    """

    def run_once(Zs_basis):
        cost, grad, hvp = (
            objective_function.compute_riemannian_loss_gradient_hvp_HST_loop(
                ket,
                bra,
                Gs,
                Zs_basis,
                qubits,
                True,
                chi_max,
                is_TI=True,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        cost.block_until_ready()
        grad.block_until_ready()
        hvp.block_until_ready()
        return hvp

    local_basis, local_basis_names = get_two_qubit_skew_hermitian_pauli_basis()
    n_local = local_basis.shape[0]
    dim = nlayers * n_local

    basis_vectors = []
    basis_vector_names = []

    for layer in range(nlayers):
        for basis_idx in range(n_local):
            vec = jnp.zeros_like(Gs)
            vec = vec.at[layer].set(local_basis[basis_idx])
            if project_basis_to_tangent:
                vec = riemannian_manifold.project_to_tangent_space(Gs, vec)
            basis_vectors.append(vec)
            basis_vector_names.append(f"layer={layer}, {local_basis_names[basis_idx]}")

    basis_vectors = jnp.asarray(basis_vectors)
    hvp_images = []
    for i, basis_vec in enumerate(basis_vectors):
        print(f"{i+1}/{len(basis_vectors)}")
        hvp_images.append(run_once(basis_vec))
    hvp_images = jnp.asarray(hvp_images)

    basis_flat = np.asarray(basis_vectors).reshape(dim, -1)
    hvp_flat = np.asarray(hvp_images).reshape(dim, -1)

    gram = np.real(basis_flat.conj() @ basis_flat.T)
    rhs = np.real(basis_flat.conj() @ hvp_flat.T)
    gram_diag = np.diag(gram)
    offdiag = gram - np.diag(gram_diag)
    offdiag_max = np.max(np.abs(offdiag))

    if offdiag_max < 1e-12:
        hessian_mat = rhs / gram_diag[:, None]
        projection_mode = "orthogonal-basis projection"
    else:
        hessian_mat = np.linalg.solve(gram, rhs)
        projection_mode = "Gram solve (non-orthogonal basis)"

    # Symmetry diagnostics:
    #  - plain symmetry is basis-dependent and only expected for orthonormal bases
    #  - metric symmetry (G H = (G H)^T) is the correct self-adjointness test
    sym_plain = np.linalg.norm(hessian_mat - hessian_mat.T) / max(
        np.linalg.norm(hessian_mat), 1e-16
    )
    gh = gram @ hessian_mat
    sym_metric = np.linalg.norm(gh - gh.T) / max(np.linalg.norm(gh), 1e-16)

    # Convert to an orthonormal basis using Gram Cholesky factor for plotting/spectrum
    chol = np.linalg.cholesky(gram)
    hessian_orth = chol @ hessian_mat @ np.linalg.inv(chol)
    hessian_orth = 0.5 * (hessian_orth + hessian_orth.T)

    basis_mode = "tangent-projected basis" if project_basis_to_tangent else "raw basis"

    print(f"Local basis size: {n_local}")
    print(f"Total basis size: {dim}")
    print(f"HVP images shape: {hvp_images.shape}")
    print("HVP mode: Riemannian HVP")
    print(f"Basis mode: {basis_mode}")
    print(f"Gram max |offdiag|: {offdiag_max:.3e}")
    print(f"Hessian assembly: {projection_mode}")
    print(f"Hessian matrix shape: {hessian_mat.shape}")
    print(f"Relative plain symmetry ||H-H^T||/||H||: {sym_plain:.3e}")
    print(f"Relative metric symmetry ||GH-(GH)^T||/||GH||: {sym_metric:.3e}")

    return hessian_mat, hessian_orth, gram, basis_vector_names


def load_checkpoint_gates(config, step):

    # ----- Load configuration -----
    config, qubits = load_preoptimized_model(config)

    # ----- Load samples -----
    initial_states, reference_states, _ = save_model.load_reference_data(
        config["sample_filename"]
    )
    initial_states = initial_states[: config["n_samples"]]
    reference_states = reference_states[
        : config["n_samples"]
    ].conj()  # Our convention for inner product
    print(
        f"Loaded sample states have bond dimensions: psi->{initial_states.shape[2]}, phi->{reference_states.shape[2]}"
    )

    # Preprocessing for loaded data
    initial_states = tn_helpers.pad_mps_to_max_bonddim(
        initial_states, config["truncation_dim"], keep_axes=True
    )
    reference_states = tn_helpers.pad_mps_to_max_bonddim(
        reference_states, config["truncation_dim"], keep_axes=True
    )

    # ----- Load checkpoint -----
    Gs, _, _, _, _, _ = save_model.load_specific_checkpoint_trust_region(
        config["load_dir"], step
    )

    print("Checkpoint loaded... ", Gs.shape)

    return Gs, config, qubits, initial_states, reference_states


def main(idx=None):
    with open(argv[1], 'r') as f:
        config = safe_load(f)
    hamiltonian = config['hamiltonian']
    config = setup_config(config)

    print(f"Performing step {idx}" if idx is not None else "")
    n_steps = 10
    if idx is None:
        iterator = range(1,n_steps+1)
    else:
        iterator = [
            idx,
        ]
        if idx > n_steps:
            return

    for step in iterator:
        if os.path.exists(f"./{hamiltonian}/spectrum/hessian_coords_{step}.npy"):
            print(f"Step {step} exists")
            continue
        print(f"step {step+1}/31")
        Gs, config, qubits, bra, ket = load_checkpoint_gates(config, step)
        hessian, hessian_orth, gram, names = get_hessian(
            qubits, ket, bra, Gs, config["truncation_dim"], config["n_layers"]
        )
        np.save(f"./{hamiltonian}/spectrum/hessian_coords_{step}.npy", hessian)
        np.save(f"./{hamiltonian}/spectrum/hessian_orth_{step}.npy", hessian_orth)
        np.save(f"./{hamiltonian}/spectrum/gram_{step}.npy", gram)
    
   # Gs, config, _, _, _ = load_checkpoint_gates(hamiltonian, 0)

    nqubits = config["n_sites"]
    if idx is not None:
        print("Done")
        return
    fig, axes = plt.subplots(1, 1, figsize=(3, 3))
    step_values = np.arange(n_steps)
    cmap = plt.get_cmap("viridis", n_steps)
    norm = colors.BoundaryNorm(np.arange(-0.5, n_steps + 0.5, 1), cmap.N)

    for step in range(n_steps):
        coords_path = f"./{hamiltonian}/spectrum/hessian_coords_{step}.npy"
        orth_path = f"./{hamiltonian}/spectrum/hessian_orth_{step}.npy"
        if not (os.path.exists(coords_path) and os.path.exists(orth_path)):
            continue

        hessian_coords = np.load(coords_path)
        hessian_orth = np.load(orth_path)

        evals_coords = np.linalg.eigvalsh(hessian_coords)
        evals_coords = np.sort(np.clip(np.abs(evals_coords.real), 1e-16, None))

        evals_orth = np.linalg.eigvalsh(hessian_orth)
        evals_orth = np.sort(np.clip(np.abs(evals_orth.real), 1e-16, None))

        evals_coords_pts, evals_orth_pts = range(len(evals_coords)), range(len(evals_orth))


        

        step_color = cmap(step)


        axes.plot(
            evals_orth_pts,
            evals_orth,
            marker="o",
            linestyle="none",
            markersize=3,
            alpha=0.6,
            color=step_color,
        )

    axes.set_title("Spectrum Riemannian Hessian")
    axes.set_xlabel("Index")
    axes.set_yscale("log")
    axes.grid(True)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(step_values)
    cbar = fig.colorbar(sm, ax=axes, ticks=np.arange(0, n_steps, 5), pad=0.02)
    cbar.set_label("Step")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fname = f"{hamiltonian}_spectrum.pdf"
    fdir = os.path.join(config['spectrum_dir'], fname)
    plt.savefig(fdir)
    plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--idx", type=int, default=None)
    # args = parser.parse_args()

    # main("ising-1d", args.idx)
    # main("heisenberg", args.idx)

    main()
    #main("heisenberg")
