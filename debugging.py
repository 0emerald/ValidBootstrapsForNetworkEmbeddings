import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from embedding_functions import *
from experiment_setup import *
import gc
from functions_for_bootstrap import parametric_bootstrap


# Generate an SBM of two communities
n = 700
T = 2
d = 2

As, tau, _ = make_iid(n, T, iid_prob=0.55)  # Easy
# As, tau, _ = make_iid(n, T, iid_prob=0.9)  # Hard

# Bootstrap first time point B times using parametric bootstrap
B = 3000

P_hat = np.random.uniform(0, 1, n**2).reshape((n, n))

# A_star = np.zeros((B, n, n))
# for b in range(B):
#     A_star = make_inhomogeneous_rg(P_hat)

# A_star = np.random.uniform(0, 1, n**2 * B).reshape((B, n, n))

# A_star = parametric_bootstrap(As[0], d, B=B)
