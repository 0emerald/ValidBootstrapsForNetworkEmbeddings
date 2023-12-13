# ResamplingAdjacencyMatrices

Here we want to take one observed A matrix, and create replicates that we believe have the same underlying generation process. We endeavour to be able to estimate properties of the graph, such as node mean and variance, from our generated matrices.

### What's been done

Have code that uses parametric bootstrap and returns a p-value distribution as to whether a new point, $\hat{\mathbf{Y}}_{i}^{(2)}$ is from the same distribution as $\hat{\mathbf{Y}}^{(1)}_i, \tilde{\mathbf{Y}}_i^{(1)}, \dots, \tilde{\mathbf{Y}}_i^{(B)}$.

    In theory it's a procedure that can define some uncertainty on each node without assuming constant variance.

This can be used to empirically verify if the procedure is valid. Currently it is not when n=500 and iid_prob=0.65. This is possibly due to the $P$ problem, where $\hat{\mathbf{P}} \notin [0,1]$.

We have a lemma that states that for [a high enough] $n$, $\hat{\mathbf{X}}$ will be sufficiently close to $\tilde{\mathbf{X}}$ such that $\hat{\mathbf{X}} \hat{\mathbf{X}}^{\top} \in [0,1]$. The condition on $n$ needs to be made concrete.

### Next steps

-   [ ] Make concrete the condition on $n$ to avoid the P matrix problem.
-   [ ] FIX BOOTSTRAP CODE AS IT DOES NOT SPAN THE TRUE OBS
-   [ ] We need to see one embedding and all it's bootstrap values, for multiple nodes, and plot also a line of its variance (which we know bc we know the distribution we sample it from)
-   [ ] Figure out why when the SBM model is more difficult (comms are closer e.g.) our bootstrapping fails.
-   [ ] Write up in maths what our bootstrap procedure is. - it is currently parametric, but we should document this properly.
-   [ ] is sample covariance an unbiased estimator of variance?
-   [ ] trying to bound something we didnt observe with something we did observe (the varaince), how do we get a bound and how is it a functoin of the thing we did observe? VI? are we systematically biased?
-   [ ] If $p$ is small or big, you have a higher chance of being pulled to or away from the origin???? variance at the origin is 0. stochastic problem, low  degree nodes we maybe underpredict their variance? Dan says there is an explanation for systematic varaince problem close to the origin
-   [ ] replace P by an estimate of the upper bound of P. can make an upper bound on degree. n=4 edges comes from some BInomial(N,p), N is number of nodes in data, p is true p. , whats a theoretical max of p, so a C.I for (1-p)p, ie the max varaince for hte binomial.
-   [ ] IN THE worst case we want ot be conservative but we defo dont want to be superuniform 
-   [ ] 
-   [ ] We should exclude self edges, $p_{ii}$ =0, convention? who cares
-   [ ] 
-   [ ] Prove that $\hat{P}$ converges to $P$ which we believe is unbiased. We need to think about bias, think that we need things to be unbiased
-   [ ] Can we go as far as to define the distribution of boostrapped samples under spectral embedding? (This will presumably look something like $\mathcal{N}(\mathbf{X}_{i},d\hat{\sigma}_{i})$ for $n$ satisfying the non-saturation condition.
-   [ ] Define some metric of how well a boostrap does.
-   [ ] Do other bootstrapping methods produce better results?
-   [ ] Apply to real data example (Emerald's data)

exchangeability_of_resampling.py: tests a given resampling method to see if its valid. I.e. resampled adjacencies produce exchangeable embeddings under UASE.
variance_estimation.py: currently uselesss...
