# Bayesian Coresets: Automated, Scalable Inference

This repository provides a python package that can be used to construct [Bayesian coresets](http://arxiv.org/abs/1710.05053). It also contains code to run (updated versions of) the experiments in [Bayesian Coreset Construction via Greedy Iterative Geodesic Ascent](https://arxiv.org/abs/1802.01737) and [Sparse Variational Inference: Bayesian Coresets from Scratch](https://arxiv.org/abs/1906.03329) in the `bayesian-coresets/examples/` folder. 

A **coreset** (i.e. the "core of a dataset") is a small, weighted subset of a dataset that can be used in place of the original dataset when learning a statistical model. If the coreset is much smaller than the original dataset, generally this makes learning faster; but if the coreset is too small, it doesn't represent the original dataset well. Building a coreset that is both small and a good approximation for the purposes of Bayesian inference is what the code in this repository does.

### Repository Status

The most recent update (Oct 2020):
- improves command line argument specification using `argparse`
- implements pseudocoresets
- incorporates weighted STAN samplers for MCMC inference

But: the repository is **no longer thoroughly tested**. Examples run and generate verified output using Python 3. Python 2 is not tested. Unit tests have not yet been updated. Work is in progress.

### Installation and Dependencies

To install with pip, download the repository and run `pip3 install . --user` in the repository's root folder. 

### Algorithms 

#### Sparse Regression

The least computationally expensive way of building a coreset is often to
discretize your data log-likelihoods and solve a sparse linear regression
problem with the discretized log-likelihood vectors.  The `bc.HilbertCoreset`
class implements these, and examples of their use can be found in the
`examples/`. The primary drawback of this sort of technique is that the user
must select/design discretization points for the log-likelihoods. The examples
in this repository achieve this by using samples from a weighting distribution
based on the Laplace posterior approximation. 

#### Sparse Variational Inference

The easiest / most automated way of building a coreset is to solve a sparse
variational inference problem. While this doesn't require any user input beyond
a generative model and dataset, it is typically much more computationally costly
than the sparse regression-based methods above. The `bc.SparseVICoreset` 
class implements this method, and examples of its use may be found in the
`examples/` folder.

#### Pseudocoresets

With very high dimensional data, it is sometimes not possible to summarize 
the data with a sparse subset. In these situations, one can sometimes still
summarize with *pseudodata*, or synthetic data. The `bc.BatchPSVICoreset` class
implements this method.

### Citations

Below are some papers to cite if you find the algorithms in this repository useful in your own research:

* D. Manousakas, Z. Xu, C. Mascolo and T. Campbell, "Bayesian pseudocoresets" Advances in Neural Information Processing Systems, 2020.
* T. Campbell and B. Beronov, "[Sparse variational inference: Bayesian coresets from scratch](https://arxiv.org/abs/1906.03329)," Advances in Neural Information Processing Systems, 2019.
* T. Campbell and T. Broderick, "[Automated scalable Bayesian inference via Hilbert coresets](https://arxiv.org/abs/1710.05053)," Journal of Machine Learning Research 20(15):1-38, 2019.
* T. Campbell and T. Broderick, "[Bayesian coreset construction via Greedy Iterative Geodesic Ascent](https://arxiv.org/abs/1802.01737)," International Conference on Machine Learning (ICML), 2018.
* J. Huggins, T. Campbell and T. Broderick, "[Coresets for scalable Bayesian logistic regression](https://arxiv.org/abs/1605.06423)," Advances in Neural Information Processing Systems, 2016.

### License Info

This code is offered under the [MIT License](https://opensource.org/licenses/MIT).
