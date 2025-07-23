# 📘 Project Overview

This repository contains code and data accompanying the paper:

**“Valid Bootstraps for Network Embeddings with Applications to Network Visualisation”**  
(*https://arxiv.org/abs/2410.20895*)

We provide implementations of our proposed network bootstrap methods, comparisons with existing approaches, and visualisations using both synthetic and real-world data.  
The notebooks reproduce the main figures from the paper and allow experimentation with different settings and bootstrap variants.

---

## 📂 Contents

### Notebooks

- `4comm_SBM_k_scores_SYMM.ipynb` - n=1000 4 community SBM, looks at Bootstrap Validity Score vs k for k=2,...,500 for ASE-kNN bootstrap and evaluates the Bootstrap Validity Score. 
- `4comm_SBM_differentBootstrapMethodsSYMM.ipynb` - applies different bootstrap methods to the n=1000, 4 comm SBM example in the Appendix and outputs the QQ-plots and Bootstrap Validity Scores. Evaluates various bootstraps on the data. 
- `MMSBM_differentBootstrapMethods_SYMM.ipynb` - creates n=300 3comm MMSBM. Evaluates various bootstraps on the data. Compares effects of n changing at the end
- `school_example_figure1code_fewClasses.ipynb` - uses classes 4A, 4B, 5A, 5B to create the plots shown in Figure 1 of paper
- `school_example.ipynb` - creates all other figures for the school data example. 


---

### 📊 Data

- `ia-primary-school-proximity-attr.edges` - data for the Lyon School social network example


---

### ⚙️ Functions

- `embedding_functions.py` - contains functions for embedding networks and code for aired displacement test
- `functions_for_bootstrap.py` - contains different functions for bootstrapping, and functions used within. Also functions for the plotting and visualisation support
- `experiment_setup.py` - functions for the setup of synthetic examples and synthetic networks

---

- ## 📌 Highlights

- All bootstrap methods can be evaluated using an **exchangeability test**, giving an empirical measure of validity.
- The proposed **kNN-based bootstrap** passes this test in cases where standard methods fail.
- Visualisation with **t-SNE** is integrated to demonstrate how uncertainty estimates can help detect spurious or misleading structure in low-dimensional embeddings.
