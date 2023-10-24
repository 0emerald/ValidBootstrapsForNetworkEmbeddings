# ResamplingAdjacencyMatrices
Here we want to take one observed A matrix, and create replicates that we believe have the same underlying generation process. We endeavour to be able to estimate properties of the graph, such as node mean and variance, from our generated matrices. 

exchangeability_of_resampling.py: tests a given resampling method to see if its valid. I.e. resampled adjacencies produce exchangeable embeddings under UASE.
variance_estimation.py: currently uselesss...
