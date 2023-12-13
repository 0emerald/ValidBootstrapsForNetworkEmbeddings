# ResamplingAdjacencyMatrices

Here we want to take one observed A matrix, and create replicates that we believe have the same underlying generation process. We endeavour to be able to estimate properties of the graph, such as node mean and variance, from our generated matrices.

### What's been done

Have code that uses parametric bootstrap and returns a p-value distribution as to whether a new point, $\hat{\mathbf{Y}}_{i}^{(2)}$ is from the same distribution as $\hat{\mathbf{Y}}^{(1)}_i, \tilde{\mathbf{Y}}_i^{(1)}, \dots, \tilde{\mathbf{Y}}_i^{(B)}$.

    In theory it's a procedure that can define some uncertainty on each node without assuming constant variance.

This can be used to empirically verify if the procedure is valid. Currently it is not when n=500 and iid_prob=0.65. This is possibly due to the P saturation problem, where $\hat{\mathbf{P}} \notin [0,1]$.

We have a lemma that states that for [a high enough] $n$, $\hat{\mathbf{X}}$ will be sufficiently close to $\tilde{\mathbf{X}}$ such that $\hat{\mathbf{X}} \hat{\mathbf{X}}^{\top} \in [0,1]$. The condition on $n$ needs to be made concrete.

### Next steps

-   [ ] Make concrete the condition on $n$ to avoid the P matrix saturation problem.
-   [ ] Can we go as far as to define the distribution of boostrapped samples under spectral embedding? (This will presumably look something like $\mathcal{N}(\mathbf{X}_{i}, d \hat{\sigma}_{i})$ for $n$ satisfying the non-saturation condition.
-   [ ] Define some metric of how well a boostrap does.
-   [ ] Do other bootstrapping methods produce better results?
-   [ ] Apply to real data example (Emerald's data)
