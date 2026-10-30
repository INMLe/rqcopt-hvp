# rqcopt-hvp
This repository accompanies: Isabel Nha Minh Le, Roeland Wiersema, and Christian B. Mendl, "Hessian-vector products for tensor networks via recursive tangent-state propagation" (2026). 


## Setup
A virtual environment for using this repository is given in ``requirements.txt``.

The implementation of methods for the Riemannian quantum circuit optimization based on matrix product states is given in ``rqcopt_mps``.
You need to install this locally to run the code: ``python3 -m pip install -e .``.


## Running Riemannian quantum circuit optimization
In this project, we optimize one-dimensional Ising and Heisenberg models.
The folder ``1_circuit-optimization`` contains ...
* ... the example configurations for each model,
* ... the script to generate the reference samples,
* ... the reference samples required to execute an exemplary optimization,
* ... the script to run the optimization using Riemannian ADAM or trust-region.

The exemplary configurations are chosen such that the execution should not take more than 15 minutes.

You can generate the reference by running ``python3 generate-reference.py hamiltonian/configs/config_reference.yml``.
Similarly, you can run the Riemannian optimization by executing ``python3 run-optimization.py hamiltonian/configs/config_{}.yml`` with ``config_tr.yml`` for the trust-region method and ``config_adam.yml`` for the Riemannian ADAM method.
In both cases, ``hamiltonian`` should be replaced by the folder name of the model you want to run, i.e., ``ising-1d`` or ``heisenberg``.

## Benchmarking the derivatives
The folder ``2_benchmark_deriv`` contains a script that measures the compile and execution time of the gradient and of the gradient plus HVP.
Run ``python3 benchmark-derivatives.py configs/config_{hamiltonian}.yml`` with ``ising`` or ``heisenberg``.
The samples are taken from ``1_circuit-optimization/{hamiltonian}/samples``; if none exist for the configured system size, random product states are used (this does not change the timings).
Results are written to ``2_benchmark_deriv/results``.

## Evaluating original Trotter circuits
The folder ``3_trotterization`` contains a script to evaluate the (not-optimized) Trotter circuits.
Run ``python3 run-trotter-comparison.py configs/config_{hamiltonian}.yml``.

## Reproducing figures from manuscript
The folder ``4_plotting-results`` contains the numerical data and the script used to generate Figs. 4-6 in the publication.
Run ``python3 plot-results.py``.


