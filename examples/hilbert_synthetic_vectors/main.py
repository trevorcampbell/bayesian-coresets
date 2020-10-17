from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
import time
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import results
import args

class IDProjector(bc.Projector):
  def update(self, wts, pts):
    pass

  def project(self, pts, grad=False):
    return pts

arguments = args.parser.parse_args()

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

results.save(arguments, err = err, csize = csize, cput = cput)

