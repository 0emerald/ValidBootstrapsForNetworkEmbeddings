# ResamplingAdjacencyMatrices
Here we want to take one observed A matrix, and create replicates that we believe have the same underlying generation process. We endeavour to be able to estimate properties of the graph, such as node mean and variance, from our generated matrices.
## To do

### What's been done

Have code that uses parametric bootstrap and returns a p-value distribution as to whether a new point, $\hat{\mathbf{Y}}_{i}^{(2)}$ is from the same distribution as $\hat{\mathbf{Y}}^{(1)}_i, \tilde{\mathbf{Y}}_i^{(1)}, \dots, \tilde{\mathbf{Y}}_i^{(B)}$.

In theory it's a procedure that can define some uncertainty on each node without assuming constant variance.

This can be used to empirically verify if the procedure is valid. Currently it is not when n=500 and iid_prob=0.65. This is possibly due to the $P$ problem, where $\hat{\mathbf{P}} \notin [0,1]$.

We have a lemma that states that for [a high enough] $n$, $\hat{\mathbf{X}}$ will be sufficiently close to $\tilde{\mathbf{X}}$ such that $\hat{\mathbf{X}} \hat{\mathbf{X}}^{\top} \in [0,1]$. The condition on $n$ needs to be made concrete.

15April2024 update: 
* Have a kNN based bootstrap method, and a test to compare how well this performs. Can think of it as a "local" bootstrap, not global like $\hat{\mathbf{P}}=\hat{\mathbf{X}}\hat{\mathbf{X}}^{\top}$.
* KL divergence to see how well the bootstraps per node approximate the distribution of the node when compared with the distribution found from many samples from the true dist, in the space of the embedding. 

## Trying to do 
Find bootstraps that work to replicate embeddings well 
A procedure that evaluates how well a bootstrap replicate follows the true underlying distribution of the observation, for the case where only a single observation is made. 
"A principled method to evaluate the validity of bootstrap replications for graphs (adjacency matrices)"
(Naturally this is made easier to evaluate with more observations - i.e. if multiple obs, we advise the reader do this)
* 
22April2024 update: 
* the `mixed_membership_bootstraps.ipynb` contains stuff on MM SBM and DCSBM. Also see functions for testing with bootstraps in the `functions_for_bootstraps.py` file, via our kNN method. 
* Example notebooks for real data, saved under sensible names too.

## new to do 
- [ ] fix plots and terminology
- [ ] explain the p-value testing procedure
- [x] DON'T HARD CODE B INTO TEST_BOOTSTRAP, FIX THIS IN ALL THE CODE THAT USES IT WHILE YOU'RE THERE
- [x] also fixed that n_neighbors had been hard coded into the test_bootstrap function
- [x] SHOULD TEST_BOOTSTRAP USE EUCLIDEAN OR MINKOWSKI DISTANCE, IF IT USES THE LATTER, WHAT IS THE VALUE OF p? COS p=2 IS EUCLIDEAN.
- [ ] ^^ Answer: the default for the function is metric='minkowski', p=2, which is Eucliean, but I hard coded these choices in so we are in fact using Euclidean.
- [ ] UPDATE TEST_BOOTSTRAP TO TAKE IN A WEIGHTED GRAPH AS WELL, PERHAPS A ARGUEMNT TO FUCNTION. tHEN NEED TO MAKE A WEIGHTED_INHOMOGENOUS_RG FUNCTION - OR SOMETHING TO THIS NAME, BUT A MODEL THAT WORKS FOR A WEIGHTED GRAPH
- [ ] This is why test_bootstrap doesn't work for hte trade data that is weighted
- [ ] CODE UP A WEIGHTED THING


## To do

-   [ ] Dive into the literature to find relevant bootstrap procedures. for matrices! preferably adjacency matrices. 
-   [ ] Plot these procedures in p-value space (some will be super-uniform, some sub-uniform, can we predict where a given procedure will land?)
-   [ ] Can we find a way of exploiting our stability test to make our own bootstrap procedure that beats the others. - this is very hard, ask Dan about this. #
-   [ ] Exchangeability is proved for a symmetric f:[0,1]^2->R (??) and I think Ian extends this to weighted things, but is it somewhere,, or must we prove it holds, for weighted and directed graphs. Note which proofs we want to leverage. 
-   [ ] Feature matrices can be seen as weighted directed matrices, but not square. No we like adjacency matrices
-   [ ] Make some nice maths definition that exchangeable embeddings from UASE, so a spectral unfolding, provides adjacency matrices we believe follow the same underlying distribution, if they satisfy our testing procedure.
-   [ ] 

-   [ ] (What cool things are people using bootstraps for in the literature - this is like a 4 line lit review task. )
