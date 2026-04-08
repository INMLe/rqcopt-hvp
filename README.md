# rqcopt-hvp
This repository accompanies: Isabel Nha Minh Le, Roeland Wiersema, and Christian B. Mendl, "Hessian-vector products for tensor networks via recursive tangent-state propagation" (2026). 


A virtual environment for using this repository is given in ``requirements.txt``.

The implementation of methods for the Riemannian quantum circuit optimization based on matrix product states is given in ``rqcopt_mps``.
You need to install this locally to run the code: ``python3 -m pip install -e .``.

In this project, we optimize one-dimensional Ising and Heisenberg models.
The folder ``1_circuit-optimization`` contains ...
* ... the example configurations to execute each model,
* ... the script to generate the references,
* ... the script to run the Riemannian optimization.

The examplary configurations are chosen such that the execution should not take more than 2 minutes.

You can generate the reference by running ``python3 generate-reference.py hamiltonian/configs/config reference.yml``.
Similarly, you can run the Riemannian optimization by executing ``python3 run-optimization.py hamiltonian/configs/config.yml``.
In both cases, ``hamiltonian`` should be replaced by the folder name of the model you want to run, i.e., ``ising-1d`` or ``heisenberg``.

The folder ``3_trotterization`` contains a script to evaluate the (not-optimized) Trotter circuits.

The folder ``4_plotting-results`` contains the numerical data and the script to generate the figures used in the publication.


