import jax.numpy as jnp
from jax import random
from rqcopt_mps import *
from scipy.linalg import rq

key = random.PRNGKey(0)

def test_rq():
    print("****** Test rq() ******")

    # Get random matrix
    random_matrix = random.normal(key, (5, 3))

    # Scipy implementation
    R1, Q1 = rq(random_matrix, mode='economic')

    # Jax implementation
    R2, Q2 = util.rq(random_matrix)

    decomposition_correct = jnp.allclose(random_matrix, R2 @ Q2)
    decompositions_the_same = all([jnp.allclose(R1, R2), jnp.allclose(Q1, Q2)])
    print("\tJax implementation gives a correct decomposition: ", decomposition_correct)
    print("\tTwo RQ decompositions are the same: ", decompositions_the_same)
    assert decomposition_correct
    assert decompositions_the_same


def main():
    test_rq()


if __name__ == "__main__":
    main()
