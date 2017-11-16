import numpy as np
import hilbertcoresets as hc
import time

n_trials = 10
Ms = np.unique(np.logspace(0., 4., 100, dtype=np.int32))
anms = ['GIGA', 'FW', 'RND']

##########################################
## Test 1: 1M 50-dimensional gaussian data
##########################################
N = 1000000
D = 50

err = np.zeros((len(anms), n_trials, Ms.shape[0]))
nfunc = np.zeros((len(anms), n_trials, Ms.shape[0]))
cput = np.zeros((len(anms), n_trials, Ms.shape[0]))
for tr in range(n_trials):
  X = np.random.randn(N, D)
  XS = X.sum(axis=0)
  for aidx, anm in enumerate(anms):
    print 'data: gauss, trial ' + str(tr+1) + '/' + str(n_trials) + ', alg: ' + anm
    alg = None
    if anm == 'GIGA':
      alg = hc.GIGA(X)
    elif anm == 'FW':
      alg = hc.FrankWolfe(X)
    else:
      alg = hc.RandomSubsampling(X) 

    for m, M in enumerate(Ms):
      t0 = time.clock()
      alg.run(M)
      tf = time.clock()
      cput[aidx, tr, m] = tf-t0 + cput[aidx, tr, m-1] if m > 0 else tf-t0
      wts = alg.weights()
      err[aidx, tr, m] = np.sqrt((((wts[:, np.newaxis]*X).sum(axis=0) - XS)**2).sum())
      nfunc[aidx, tr, m] = alg.get_num_ops()

np.savez_compressed('gauss_results.npz', err=err, nfunc=nfunc, cput=cput, Ms = Ms, anms=anms)

##########################################
## Test 2: 5K axis-aligned data
##########################################
 
N = 5000

X = np.eye(N)
XS = np.ones(N)

err = np.zeros((len(anms), n_trials, Ms.shape[0]))
nfunc = np.zeros((len(anms), n_trials, Ms.shape[0]))
cput = np.zeros((len(anms), n_trials, Ms.shape[0]))
for tr in range(n_trials):
  for aidx, anm in enumerate(anms):
    print 'data: axis, trial ' + str(tr+1) + '/' + str(n_trials) + ', alg: ' + anm
    alg = None
    if anm == 'GIGA':
      alg = hc.GIGA(X)
    elif anm == 'FW':
      alg = hc.FrankWolfe(X)
    else:
      alg = hc.RandomSubsampling(X) 

    for m, M in enumerate(Ms):
      t0 = time.clock()
      alg.run(M)
      tf = time.clock()
      cput[aidx, tr, m] = tf-t0 + cput[aidx, tr, m-1] if m > 0 else tf-t0
      wts = alg.weights()
      err[aidx, tr, m] = np.sqrt((((wts[:, np.newaxis]*X).sum(axis=0) - XS)**2).sum())
      nfunc[aidx, tr, m] = alg.get_num_ops()

np.savez_compressed('axis_results.npz', err=err, nfunc=nfunc, cput=cput, Ms = Ms, anms=anms)

