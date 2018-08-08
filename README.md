# Bayesian Coresets: Automated, Scalable Inference

This repository provides a python package that can be used to construct [Bayesian coresets](http://arxiv.org/abs/1710.05053). It also contains all the code used to run the experiments in [Bayesian Coreset Construction via Greedy Iterative Geodesic Ascent](https://arxiv.org/abs/1802.01737) in the `bayesian-coresets/examples/` folder.

A **coreset** (i.e. the "core of a dataset") is a small, weighted subset of a dataset that can be used in place of the original dataset when learning a statistical model. If the coreset is much smaller than the original dataset, generally this makes learning faster; but if the coreset is too small, it doesn't represent the original dataset well. Building a coreset that is both small and a good approximation for the purposes of Bayesian inference is what the code in this repository does.

### Installation and Dependencies

To install with pip, download the repository and run `pip install .` in the repository's root folder.
If you are using python 3.x, run `pip3 install .` instead. Note: this package depends on [NumPy](http://www.numpy.org) and [SciPy](https://www.scipy.org).
The examples also depend on [Bokeh](https://bokeh.pydata.org/en/latest) for plotting.

### Repository Status

The code for this package has been tested and is compatible with both python 2.7 and 3.5.

Unit tests are written for `nose`. To run the tests, install the package, navigate to the `tests` subfolder, and run `nosetests`.

All unit tests currently pass. 

### Example Usage: Bayesian Logistic Regression

The code to run this example may be found in `examples/simple_logistic_regression/`.

**Setup:** In Bayesian logistic regression, we have a dataset `x` of `N` input vectors `x[n, :]` in `D` dimensions along with `N` responses `y[n]` equal to -1 or 1, and we want to predict the response at an arbitrary input. The model is that there is a latent `D`-dimensional parameter `theta` such that `y[n] | theta ~ Bernoulli(1/(1+np.exp(-np.dot(theta, x[n, :]))))` independently across the data. We take a Bayesian approach to learning `theta`, and place a standard prior on it: `theta ~ Normal(0, I)`. When `N` is large, MCMC and variational inference run slowly; instead, we will first "compress" / "summarize" the dataset by building a coreset, and then run inference on that.

**Step 1 - Define the Model:**

**Step 2 - Obtain a Cheap Posterior Approximation:**

**Step 3 - Discretize the Log-Likelihood Functions:**

**Step 4 - Build the Coreset**

**Step 5 - Run Posterior Inference**




**Inputs:** A Bayesian model for conditionally independent data given a latent parameter (see the **Background** section below), and a dataset. To build a coreset, you need to implement two things: the gradient of the log-likelihood function, and a cheap approximate posterior sampler.
```
#dataset is an N x D numpy array, one row for each data point
#param is a length-K numpy array
#k is an integer from 1 to K
#output: grads is a length-N numpy array
def grad_log_likelihood(dataset, param, k):
   grads = [gradient of the log_likelihood with respect to dimension k of param for each datapoint in dataset]
   return grads
```
And second, an approximate posterior sampler:
```
```

If we have `N` datapoints `x_1, x_2, ..., x_N`, when we run inference (MCMC, variational inference, etc), we have to compute the joint probability `p(x_1, x_2, ... x_N, theta)` for all the data many times:
```
def log_joint(dataset, theta):
   lj = log_prior(theta)
   for i in range(N):
     lj += log_likelihood(dataset[i], theta)
   return lj
```
**Outputs:** A set of weights 


```
nonzero_idcs = [i for i in range(N) if weights[i] > 0]
...
def coreset_log_joint(dataset, weights, nonzero_idcs, theta):
   lj = log_prior(theta)
   for i in nonzero_idcs:
     lj += weights[i]*log_likelihood(dataset[i], theta)
   return lj
```

This is expensive when there are lots of datapoints, i.e., `N` is large. Rather than computing the full log joint repeatedly, instead we compute the log joint for a **Bayesian coreset:**


If the number of nonzero entries in `weights` is small compared to `N`, this function is much less expensive to compute than `log_joint`. This repository finds a good set of `weights` given a `dataset` and a Bayesian model, consisting of functions `log_likelihood`, `log_prior`, and their gradients.

**Step 1: Project**

The first step to build a coreset

**Step 2: Find Sparse Weights**



**Step 3: Posterior Inference**


### Background
In the setting of Bayesian inference, we have a model consisting of two components: a prior distribution on some latent parameter of interest `p(theta)`, and a likelihood distribution `p(x|theta)` that governs how the data are generated given the parameter `theta`:
```
def log_prior(theta):
   lp = [ compute log(p(theta)) given parameter theta ]
   return lp
   
def log_likelihood(x, theta):
   ll = [compute log(p(x|theta)) for datapoint x with parameter theta]
   return ll
```
If we have `N` datapoints `x_1, x_2, ..., x_N`, when we run inference (MCMC, variational inference, etc), we have to compute the joint probability `p(x_1, x_2, ... x_N, theta)` for all the data many times:
```
def log_joint(dataset, theta):
   lj = log_prior(theta)
   for i in range(N):
     lj += log_likelihood(dataset[i], theta)
   return lj
```
This is expensive when there are lots of datapoints, i.e., `N` is large. Rather than computing the full log joint repeatedly, instead we compute the log joint for a **Bayesian coreset:**
```
nonzero_idcs = [i for i in range(N) if weights[i] > 0]
...
def coreset_log_joint(dataset, weights, nonzero_idcs, theta):
   lj = log_prior(theta)
   for i in nonzero_idcs:
     lj += weights[i]*log_likelihood(dataset[i], theta)
   return lj
```
If the number of nonzero entries in `weights` is small compared to `N`, this function is much less expensive to compute than `log_joint`. This repository finds a good set of `weights` given a `dataset` and a Bayesian model, consisting of functions `log_likelihood`, `log_prior`, and their gradients.

### Citations

Below are some papers to cite if you find the algorithms in this repository useful in your own research:

* T. Campbell and T. Broderick, "[Automated scalable Bayesian inference via Hilbert coresets](http://arxiv.org/abs/1710.05053)," arXiv:1710.05053, 2017,
* T. Campbell and T. Broderick, "[Bayesian coreset construction via Greedy Iterative Geodesic Ascent](https://arxiv.org/abs/1802.01737)," International Conference on Machine Learning (ICML), 2018.

### License Info

This code is offered under the [MIT License](https://opensource.org/licenses/MIT).
