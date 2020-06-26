from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('tr', type=int, help='The trial number (used to seed random number generation)')
parser.add_argument('alg_nm', type=str, choices=['FW', 'GIGA', 'OMP', 'IS', 'US'], help="The sparse non negative least squares algorithm to use: one of FW (Frank Wolfe), GIGA (Greedy Iterative Geodeic Ascent), OMP (Orthogonal Matching Pursuit), IS (Importance Sampling), US (Uniform Sampling)")
parser.add_argument('--N', type=int, default=10000, help="The number of synthetic data points (only if the --diag flag is not provided)")
parser.add_argument('--d', type=int, default=100, help="The dimension of the synthetic data points (if the --diag flag is provided, this is also the number of synthetic data points")
parser.add_argument('--diag', action='store_const', default=False, const=True, help="If this flag is provided, uses an axis-aligned diagonal dataset (NxN) instead of the usual random Nxd matrix")
parser.add_argument('--fldr', type=str, default="results/", help="This script will save results in this folder. Default \"results/\"")


arguments = parser.parse_args()
tr = arguments.tr
alg_nm = arguments.alg_nm
N = arguments.N
d = arguments.d
diag = arguments.diag
fldr = arguments.fldr

class IDProjector(bc.Projector):

  def update(self, wts, pts):
    pass

  def project(self, pts, grad=False):
    return pts

np.random.seed(3)

bc.util.set_verbosity('error')

Ms = np.unique(np.logspace(0., 4., 100, dtype=np.int32))

algs = {'FW': bc.snnls.FrankWolfe, 
        'GIGA': bc.snnls.GIGA,
        'OMP': bc.snnls.OrthoPursuit, 
        'IS': bc.snnls.ImportanceSampling, 
        'US': bc.snnls.UniformSampling}

##########################################
## Test 1: gaussian data
##########################################

#######################################
#######################################
## Step 0: Generate a Synthetic Dataset
#######################################
#######################################

if diag:
  X = np.eye(d)
else: 
  X = np.random.randn(N, d)

############################
############################
## Step 1: Build/Evaluate the Coreset
############################
############################
err = np.zeros(Ms.shape[0])
opt_err = np.zeros(Ms.shape[0])
csize = np.zeros(Ms.shape[0])
cput = np.zeros(Ms.shape[0])

print('data: gauss, trial ' + str(tr) + ', alg: ' + alg_nm)
alg = bc.HilbertCoreset(X, IDProjector(), snnls = algs[alg_nm])

for m, M in enumerate(Ms):
  t0 = time.time()
  itrs = (Ms[m] if m == 0 else Ms[m] - Ms[m-1])
  alg.build(itrs) #no explicit bound on size, just run correct # iterations (size will be upper bounded by # itrs)
  tf = time.time()
  cput[m] = tf-t0 + cput[m-1] if m > 0 else tf-t0
  wts, pts, idcs = alg.get()
  csize[m] = (wts > 0).sum()
  err[m] = alg.error()

############################
############################
## Step 2: Save Results
############################
############################
if not os.path.exists(fldr):
  os.mkdir(fldr)
np.savez_compressed(os.path.join(fldr, 'gauss_results'+('_diag'if diag else '')+'_alg='+alg_nm+'_tr='+str(tr)+'_N='+str(N)+'_d='+str(d)+'.npz'), err=err, csize=csize, cput=cput, Ms = Ms, alg=alg, tr=tr)