# import pytest
# from functions_for_bootstrap import *


"""
To do

* paramteric bootstrap
- [ ] Make sure that P_hat is [0,1]^nxn

* make_iid
- [ ] Sparse option returns a list of sparse matrices
- [ ] Non sparse option is dense

* testing function
- [ ] Make sure that it's uniform when graphs are true resamples

"""


# def test_parametric_bootstrap():
#     """
#     Test parametric bootstrap
#     """
#     # Generate a random probability matrix
#     n = 700
#     d = 2
#     P = np.random.uniform(0, 1, n**2).reshape((n, n))
#     P = (P + P.T) / 2

#     # Generate a random adjacency matrix
#     A = make_inhomogeneous_rg(P)

#     # Generate B bootstrapped adjacency matrices
#     B = 100
#     A_star, P_hat = parametric_bootstrap(A, d, B, return_P_hat=True)

#     # Check that P_hat is valid
#     assert np.min(P_hat) >= 0
#     assert np.max(P_hat) <= 1
