# Bayesian Coresets: automated, scalable inference

This repository contains the main package used to construct [Bayesian coresets](http://arxiv.org/abs/1710.05053). It also contains all the code used to run the experiments in [Bayesian Coreset Construction via Greedy Iterative Geodesic Ascent](https://arxiv.org/abs/1802.01737). More details will be added soon.


### Installation and Dependencies

To install with pip, download the repository and run `pip install .` in the repository's root folder.
If you are using python 3.x, run `pip3 install .` instead. Note: this package depends on `numpy`, and the examples depend additionally on `scipy`.

### Repository Status

The code for this package has been tested and is compatible with both python 2.7 and 3.5.

Unit tests are written for `nose`. To run the tests, install the package, navigate to the `tests` subfolder, and run `nosetests`.

Unit tests for Frank--Wolfe, GIGA, and subsampling methods all currently pass. Random projections code is not yet tested.

More details about usage, output, status, etc will be updated here shortly.

### License Info

This code is offered under the [MIT License](https://opensource.org/licenses/MIT).
