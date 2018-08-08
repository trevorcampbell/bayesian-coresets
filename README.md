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

### Citations

Below are some papers to cite if you find the algorithms in this repository useful in your own research:

* T. Campbell and T. Broderick, "[Automated scalable Bayesian inference via Hilbert coresets](http://arxiv.org/abs/1710.05053)," arXiv:1710.05053, 2017,
* T. Campbell and T. Broderick, "[Bayesian coreset construction via Greedy Iterative Geodesic Ascent](https://arxiv.org/abs/1802.01737)," International Conference on Machine Learning (ICML), 2018.

### License Info

This code is offered under the [MIT License](https://opensource.org/licenses/MIT).
