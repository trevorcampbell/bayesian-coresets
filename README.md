# Bayesian Coresets: automated, scalable inference

This repository contains the main package used to construct [Bayesian coresets](http://arxiv.org/abs/1710.05053). It also contains all the code used to run the experiments in [Bayesian Coreset Construction via Greedy Iterative Geodesic Ascent](https://arxiv.org/abs/1802.01737). More details will be added soon.


### Installation and Dependencies

To install with pip, download the repository and run the following command in the repository's root folder:

    pip install . 
    
This package depends on `numpy`. The examples depend additionally on `scipy`.

### Repository Status

This package is written for Python2.7, and is untested for Python3. The package will be updated to be Python3 compatible shortly.

Unit tests are written for `nose`. To run the tests, install the package, navigate to the `tests` subfolder, and run `nosetests`.

Unit tests for Frank--Wolfe, GIGA, and subsampling methods all currently pass. Random projections code is not yet tested.

More details about usage, output, status, etc will be updated here shortly.

### License Info

This code is offered under the [MIT License](https://opensource.org/licenses/MIT).
