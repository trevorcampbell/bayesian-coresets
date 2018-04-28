from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
import time

n_trials = 20
Ms = np.unique(np.logspace(0., 4., 100, dtype=np.int32))
anms = ['GIGA', 'FW', 'MP', 'FSW', 'OMP', 'LAR', 'IS', 'RND']
algs = [bc.GIGA2, bc.FrankWolfe2, bc.Pursuit, bc.ForwardStagewise, bc.OrthoPursuit2, bc.LAR, bc.ImportanceSampling, bc.RandomSubsampling]

##########################################
## Test 1: gaussian data
##########################################
N = 1000000
D = 50

err = np.zeros((len(anms), n_trials, Ms.shape[0]))
csize = np.zeros((len(anms), n_trials, Ms.shape[0]))
cput = np.zeros((len(anms), n_trials, Ms.shape[0]))
for tr in range(n_trials):
  X = np.random.randn(N, D)
  XS = X.sum(axis=0)
  for aidx, anm in enumerate(anms):
    print('data: gauss, trial ' + str(tr+1) + '/' + str(n_trials) + ', alg: ' + anm)
    alg = algs[aidx](X)

    for m, M in enumerate(Ms):
      t0 = time.time()
      alg.run(M)
      tf = time.time()
      cput[aidx, tr, m] = tf-t0 + cput[aidx, tr, m-1] if m > 0 else tf-t0
      wts = alg.weights()
      err[aidx, tr, m] = np.sqrt((((wts[:, np.newaxis]*X).sum(axis=0) - XS)**2).sum())
      csize[aidx, tr, m] = (wts > 0).sum()

np.savez_compressed('gauss_results.npz', err=err, csize=csize, cput=cput, Ms = Ms, anms=anms)

##########################################
## Test 2: axis-aligned data
##########################################
 
N = 5000

X = np.eye(N)
XS = np.ones(N)

err = np.zeros((len(anms), n_trials, Ms.shape[0]))
csize = np.zeros((len(anms), n_trials, Ms.shape[0]))
cput = np.zeros((len(anms), n_trials, Ms.shape[0]))
for tr in range(n_trials):
  for aidx, anm in enumerate(anms):
    print('data: axis, trial ' + str(tr+1) + '/' + str(n_trials) + ', alg: ' + anm)
    alg = None
    if anm == 'GIGA':
      alg = bc.GIGA(X)
    elif anm == 'FW':
      alg = bc.FrankWolfe(X)
    else:
      alg = bc.RandomSubsampling(X) 

    for m, M in enumerate(Ms):
      t0 = time.time()
      alg.run(M)
      tf = time.time()
      cput[aidx, tr, m] = tf-t0 + cput[aidx, tr, m-1] if m > 0 else tf-t0
      wts = alg.weights()
      err[aidx, tr, m] = np.sqrt((((wts[:, np.newaxis]*X).sum(axis=0) - XS)**2).sum())
      csize[aidx, tr, m] = (wts>0).sum()

np.savez_compressed('axis_results.npz', err=err, csize=csize, cput=cput, Ms = Ms, anms=anms)

