# Bayesian Coresets: Automated, Scalable Inference

This repository contains the main package used to construct [Bayesian coresets](http://arxiv.org/abs/1710.05053). It also contains all the code used to run the experiments in [Bayesian Coreset Construction via Greedy Iterative Geodesic Ascent](https://arxiv.org/abs/1802.01737). More details will be added soon.


### Installation and Dependencies

To install with pip, download the repository and run `pip install .` in the repository's root folder.
If you are using python 3.x, run `pip3 install .` instead. Note: this package depends on [NumPy](http://www.numpy.org) and [SciPy](https://www.scipy.org).
The examples also depend on [Bokeh](https://bokeh.pydata.org/en/latest) for plotting.

### Repository Status

The code for this package has been tested and is compatible with both python 2.7 and 3.5.

Unit tests are written for `nose`. To run the tests, install the package, navigate to the `tests` subfolder, and run `nosetests`.

Unit tests for all coreset construction algorithms currently pass. Random projections code is not yet tested.

### Usage

More details about usage, output, etc will be updated here shortly.

### Citations

Below are some papers to cite if you find the algorithms in this repository useful in your own research:

* T. Campbell and T. Broderick, "[Automated scalable Bayesian inference via Hilbert coresets](http://arxiv.org/abs/1710.05053)," arXiv:1710.05053, 2017,
* T. Campbell and T. Broderick, "[Bayesian coreset construction via Greedy Iterative Geodesic Ascent](https://arxiv.org/abs/1802.01737)," arXiv:1802.01737, 2018.

### License Info

This code is offered under the [MIT License](https://opensource.org/licenses/MIT).
