### Notebooks

`4comm_SBM_k_scores_SYMM.ipynb` - n=1000 4 community SBM, looks at Bootstrap Validity Score vs k for k=2,...,500 for ASE-kNN bootstrap
`4comm_SBM_example_differentBootstrapMethodsSYMM.ipynb` - applies different bootstrap methods to the 4 comm SBM example in the Appendix and outputs the QQ-plots and Bootstrap Validity Scores
`MMSBM_differentBootstrapMethods_SYMM.ipynb` - creates n=300 3comm MMSBM. Evaluates various bootstraps on the data. Compares effects of n changing at the end
`4comm_SBM_differentBootstrapMethodsSYMM.ipynb`- creates n=1000 4comm SBM. Evaluates various bootstraps on the data. 
`4comm_SBM_k_scores_SYMM.ipynb` - for SBM example, goes through different values of k and evaluates the Bootstrap Validity Score. 
`school_example_figure1code_fewClasses.ipynb` - uses classes 4A, 4B, 5A, 5B to create the plots shown in Figure 1 of paper


### Data
`ia-primary-school-proximity-attr.edges` - data for the Lyon School social network example


### Functions
`embedding_functions.py` - contains functions for embedding networks and code for aired displacement test
`functions_for_bootstrap.py` - contains different functions for bootstrapping, and functions used within. Also functions for the plotting and visualisation support
`experiment_setup.py` - functions for the setup of synthetic examples and synthetic networks
