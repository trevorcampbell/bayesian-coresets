import numpy as np

opt_itrs = 1000
M = 20
N = 20
N_monte_carlo = 1000
th0 = np.zeros(2)
Sig0 = np.eye(2)
Sig = np.eye(2)
th = np.ones(2)
x = np.random.multivariate_normal(th, Sig, N)
Sig0inv = np.linalg.inv(Sig0)
Siginv = np.linalg.inv(Sig)
mup, Sigp = post(th0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
Sigpinv = np.linalg.inv(Sigp)

if not seeded_algs and seeded_data:
  np.random.seed()


