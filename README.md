# Bayesian Coresets: Automated, Scalable Inference

This repository provides a python package that can be used to construct [Bayesian coresets](http://arxiv.org/abs/1710.05053). It also contains all the code used to run the experiments in [Bayesian Coreset Construction via Greedy Iterative Geodesic Ascent](https://arxiv.org/abs/1802.01737) and [Sparse Variational Inference: Bayesian Coresets from Scratch](https://arxiv.org/abs/1906.03329) in the `bayesian-coresets/examples/` folder.

A **coreset** (i.e. the "core of a dataset") is a small, weighted subset of a dataset that can be used in place of the original dataset when learning a statistical model. If the coreset is much smaller than the original dataset, generally this makes learning faster; but if the coreset is too small, it doesn't represent the original dataset well. Building a coreset that is both small and a good approximation for the purposes of Bayesian inference is what the code in this repository does.

### Repository Status

After the recent update (June 2019) implementing [Sparse Variational Inference](https://arxiv.org/abs/1906.03329), the repository is **no longer thoroughly tested**. Examples run and generate verified output using Python 3. Python 2 is not tested. Unit tests have not yet been updated. Work is in progress.

### Installation and Dependencies

To install with pip, download the repository and run `pip3 install . --user` in the repository's root folder.
This code has not yet been verified with Python 2, so use `pip` at your own risk. Note: this package depends on [NumPy](http://www.numpy.org), [SciPy](https://www.scipy.org), and [SciKit Learn](https://scikit-learn.org).
The examples also depend on [Bokeh](https://bokeh.pydata.org/en/latest) for plotting.

### Examples - Sparse Regression

The simplest way of doing coresets is via sparse regression. Come up with some vec rep of your likelihood then SR.

#### Example 0: Synthetic Vectors, Sparse Regression

In this example, we build a coreset abstract vecs. The code to follow along with this example may be found in `examples/synthetic_vectors/`. Calling `python main.py` runs the example code and outputs a coreset size and posterior mean.

#### Example 1: Bayesian Logistic Regression, Sparse Regression

The code to follow along with this example may be found in `examples/simple_logistic_regression/`. Calling `python main.py` runs the example code and outputs a coreset size and posterior mean.

**Setup:** In Bayesian logistic regression, we have a dataset `x` of `N` input vectors `x[n, :]` in `D` dimensions along with `N` responses `y[n]` equal to -1 or 1, and we want to predict the response at an arbitrary input. The model is that there is a latent `D`-dimensional parameter `theta` such that `y[n] | theta ~ Bernoulli(1/(1+np.exp(-np.dot(theta, x[n, :]))))` independently across the data. We take a Bayesian approach to learning `theta`, and place a standard prior on it: `theta ~ Normal(0, I)`. When `N` is large, MCMC and variational inference run slowly; instead, we will first "compress" / "summarize" the dataset by building a coreset, and then run inference on that.

**Step 0 - Obtain/Generate Data:** In the example, we generate synthetic data.
```
#10,000 datapoints, 10-dimensional
N = 10000
D = 10
#generate input vectors from standard normal
mu = np.zeros(D)
cov = np.eye(D)
X = np.random.multivariate_normal(mu, cov, N)
#set the true parameter to [3,3,3,..3]
th = 3.*np.ones(D)
#generate responses given inputs
ps = 1.0/(1.0+np.exp(-(X*th).sum(axis=1)))
y =(np.random.rand(N) <= ps).astype(int)
#format data for (grad/hess) log (likelihood/prior/joint)
Z = y[:, np.newaxis]*X

```

**Step 1 - Define the Model:** The Bayesian logistic regression model, including log-prior/likelihood/joint functions and their derivatives, is defined in  `examples/simple_logistic_regression/model.py`. 
```
from model import *
```

**Step 2 - Obtain a Cheap Posterior Approximation:** We use the Laplace approximation to find a cheap Gaussian approximation to the posterior.
```
#first, optimize the log joint to find the mode:
res = minimize(lambda mu : -log_joint(Z, mu, np.ones(Z.shape[0])), Z.mean(axis=0), jac=lambda mu : -grad_log_joint(Z, mu, np.ones(Z.shape[0])))
#then find a quadratic expansion around the mode, and assume the distribution is Gaussian
cov = -np.linalg.inv(hess_log_joint(Z, res.x))

#we can call post_approx() to sample from the approximate posterior
post_approx = lambda : np.random.multivariate_normal(res.x, cov)
```

**Step 3 - Discretize the Log-Likelihood Functions:** The coreset construction algorithms in this repository require a finite-dimensional approximation of the log-likelihood functions for each datapoint.  
```
projection_dim = 500 #random projection dimension, K
#build the discretization of all the log-likelihoods based on random projection
proj = bc.ProjectionF(Z, grad_log_likelihood, projection_dim, post_approx) 
#construct the N x K discretized log-likelihood matrix; each row represents the discretized LL func for one datapoint
vecs = proj.get()
```

**Step 4 - Build the Coreset:** GIGA takes the discretized log-likelihood functions, and finds a sparse weighted subset that approximates the total log-likelihood for all the data.
```
M = 100 # use 100 datapoints
giga = bc.GIGA(vecs) #do coreset construction using the discretized log-likelihood functions
giga.run(M) #build the coreset
wts = giga.weights() #get the output weights
idcs = wts > 0 #pull out the indices of datapoints that were included in the coreset
```

### Examples - Sparse Variational Inference

Can also build coresets via sparse VI

#### Example 1: Bayesian Gaussian Model, Sparse Variational Inference

This is one of the examples from [Sparse Variational Inference: Bayesian Coresets from Scratch](https://arxiv.org/abs/1906.03329). The code to follow along with this example may be found in `examples/riemann_gaussian/`. Calling `python main.py` runs the example code and outputs a coreset size and posterior mean.




#### Example 4: Bayesian Logistic Regression, Sparse Variational Inference

[Sparse Variational Inference: Bayesian Coresets from Scratch](https://arxiv.org/abs/1906.03329)

### Citations

Below are some papers to cite if you find the algorithms in this repository useful in your own research:


* T. Campbell and B. Beronov, "[Sparse variational inference: Bayesian coresets from scratch](https://arxiv.org/abs/1906.03329)," arXiv:1906.03329.
* T. Campbell and T. Broderick, "[Automated scalable Bayesian inference via Hilbert coresets](https://arxiv.org/abs/1710.05053)," Journal of Machine Learning Research 20(15):1-38, 2019.
* T. Campbell and T. Broderick, "[Bayesian coreset construction via Greedy Iterative Geodesic Ascent](https://arxiv.org/abs/1802.01737)," International Conference on Machine Learning (ICML), 2018.
* J. Huggins, T. Campbell and T. Broderick, "[Coresets for scalable Bayesian logistic regression](https://arxiv.org/abs/1605.06423)," Advances in Neural Information Processing Systems, 2016.

### License Info

This code is offered under the [MIT License](https://opensource.org/licenses/MIT).
