# %%
import numpy as np
import matplotlib.pyplot as plt
from embedding_functions import *
from experiment_setup import *

# %%

min_emb = []
max_emb = []
for n in [100, 200, 500, 1000]:
    B = [np.array([[0.5, 0.2], [0.2, 0.5]])]
    As, tau, _ = sbm_from_B(n=n, Bs=B, return_p=True)

    ya = single_spectral(As[0], 2)

    p_hat = ya @ ya.T

    print(np.max(p_hat))
    print(np.min(p_hat))

#     min_emb.append(np.min(p_hat))
#     max_emb.append(np.max(p_hat))

# plt.figure()
# plt.plot(max_emb)
# plt.plot(min_emb)

# %%
