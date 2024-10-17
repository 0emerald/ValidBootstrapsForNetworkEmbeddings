### Notebooks

- `4comm_SBM_k_scores_SYMM.ipynb` - n=1000 4 community SBM, looks at Bootstrap Validity Score vs k for k=2,...,500 for ASE-kNN bootstrap and evaluates the Bootstrap Validity Score. 
- `4comm_SBM_differentBootstrapMethodsSYMM.ipynb` - applies different bootstrap methods to the n=1000, 4 comm SBM example in the Appendix and outputs the QQ-plots and Bootstrap Validity Scores. Evaluates various bootstraps on the data. 
- `MMSBM_differentBootstrapMethods_SYMM.ipynb` - creates n=300 3comm MMSBM. Evaluates various bootstraps on the data. Compares effects of n changing at the end
- `school_example_figure1code_fewClasses.ipynb` - uses classes 4A, 4B, 5A, 5B to create the plots shown in Figure 1 of paper
- `school_example.ipynb` - creates all other figures for the school data example. 


### Data
- `ia-primary-school-proximity-attr.edges` - data for the Lyon School social network example


### Functions
- `embedding_functions.py` - contains functions for embedding networks and code for aired displacement test
- `functions_for_bootstrap.py` - contains different functions for bootstrapping, and functions used within. Also functions for the plotting and visualisation support
- `experiment_setup.py` - functions for the setup of synthetic examples and synthetic networks
