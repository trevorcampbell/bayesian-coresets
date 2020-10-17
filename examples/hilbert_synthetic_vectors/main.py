from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
import time
import argparse
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import results


class IDProjector(bc.Projector):
  def update(self, wts, pts):
    pass

  def project(self, pts, grad=False):
    return pts

##########################################
### Setup Argument Parser
##########################################
parser = argparse.ArgumentParser()

# example-specific arguments
parser.add_argument('alg_nm', type=str, choices=['FW', 'GIGA', 'OMP', 'IS', 'US'], help="The sparse non negative least squares algorithm to use: one of FW (Frank Wolfe), GIGA (Greedy Iterative Geodeic Ascent), OMP (Orthogonal Matching Pursuit), IS (Importance Sampling), US (Uniform Sampling)")
parser.add_argument('--data_num', type=int, default=1000, help="The number of synthetic data points")
parser.add_argument('--data_dim', type=int, default=100, help="The dimension of the synthetic data points, if applicable")
parser.add_argument('--data_type', type=str, default='normal', choices=['normal', 'axis'], help="Specifies the type of synthetic data to generate.")
parser.add_argument('--coreset_size_max', type=int, default=1000, help="The maximum coreset size to evaluate")
parser.add_argument('--coreset_num_sizes', type=int, default=100, help="The number of coreset sizes to evaluate")
parser.add_argument('--coreset_size_spacing', type=str, choices=['log', 'linear'], default='log', help="The spacing of coreset sizes to test")

# common arguments
parser.add_argument('--trial', type=int, help='The trial number (used to seed random number generation)')
parser.add_argument('--results_folder', type=str, default="results/", help="This script will save results in this folder. Default \"results/\"")
parser.add_argument('--verbosity', type=str, default="error", choices=['error', 'warning', 'critical', 'info', 'debug'], help="The verbosity level.")

# parse
arguments = parser.parse_args()

# check if result already exists for this run, and if so, quit
if results.check_exists(arguments):
  print('Results already exist for arguments ' + str(arguments))
  print('Quitting.')
  quit()

#######################################
#######################################
########### Step 0: Setup #############
#######################################
#######################################

np.random.seed(arguments.trial)
bc.util.set_verbosity(arguments.verbosity)
algs = {'FW': bc.snnls.FrankWolfe, 
        'GIGA': bc.snnls.GIGA,
        'OMP': bc.snnls.OrthoPursuit, 
        'IS': bc.snnls.ImportanceSampling, 
        'US': bc.snnls.UniformSampling}

if arguments.coreset_size_spacing == 'log':
    Ms = np.unique(np.logspace(0., np.log10(arguments.coreset_size_max), arguments.coreset_num_sizes, dtype=np.int32))
else:
    Ms = np.unique(np.linspace(1, arguments.coreset_size_max, arguments.coreset_num_sizes, dtype=np.int32))

#######################################
#######################################
## Step 1: Generate a Synthetic Dataset
#######################################
#######################################

if arguments.data_type == 'normal':
  X = np.random.randn(arguments.data_num, arguments.data_dim)
else: 
  X = np.eye(arguments.data_num)

############################
############################
## Step 1: Build/Evaluate the Coreset
############################
############################

data_type = arguments.data_type
fldr = arguments.results_folder

err = np.zeros(Ms.shape[0])
csize = np.zeros(Ms.shape[0])
cput = np.zeros(Ms.shape[0])

print('data: ' + arguments.data_type + ', trial ' + str(arguments.trial) + ', alg: ' + arguments.alg_nm)
alg = bc.HilbertCoreset(X, IDProjector(), snnls = algs[arguments.alg_nm])

for m, M in enumerate(Ms):
  t0 = time.process_time()
  itrs = (Ms[m] if m == 0 else Ms[m] - Ms[m-1])
  alg.build(itrs) 
  tf = time.process_time()
  cput[m] = tf-t0 + cput[m-1] if m > 0 else tf-t0
  wts, pts, idcs = alg.get()
  csize[m] = (wts > 0).sum()
  err[m] = alg.error()

############################
############################
## Step 2: Save Results
############################
############################

results.save_result(arguments, err = err, csize = csize, cput = cput)

