from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
import time
import os
import sys
import argparse
import bokeh.plotting as bkp
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import results
import plotting

class IDProjector(bc.Projector):
  def update(self, wts, pts):
    pass

  def project(self, pts, grad=False):
    return pts

def plot(arguments):
    # extract the non-plotting-related arguments that the user specified (we will use these to match on when loading results)
    dargs = vars(arguments)
    matching_dict = {anm : dargs[anm] for anm in args.argnames if dargs[anm] is not None}
    # load only the results that match (avoid high mem usage)
    resdf = results.load_matching(matching_dict)
    
    #call the generic plot function
    plotting.plot(arguments, resdf)

def run(arguments):
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
    
    results.save(arguments, err = err, csize = csize, Ms = Ms, cput = cput)


############################
############################
## Parse arguments
############################
############################
 

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='sub-command help')
run_subparser = subparsers.add_parser('run', help='Runs the main computational code')
run_subparser.set_defaults(func=run)
plot_subparser = subparsers.add_parser('plot', help='Plots the results')
plot_subparser.set_defaults(func=plot)

# example-specific arguments
run_subparser.add_argument('--alg_nm', type=str, default='GIGA', choices=['FW', 'GIGA', 'OMP', 'IS', 'US'], help="The sparse non negative least squares algorithm to use")
run_subparser.add_argument('--data_num', type=int, default=10000, help="The number of synthetic data points")
run_subparser.add_argument('--data_dim', type=int, default=100, help="The dimension of the synthetic data points, if applicable")
run_subparser.add_argument('--data_type', type=str, default='normal', choices=['normal', 'axis'], help="Specifies the type of synthetic data to generate.")
run_subparser.add_argument('--coreset_size_max', type=int, default=1000, help="The maximum coreset size to evaluate")
run_subparser.add_argument('--coreset_num_sizes', type=int, default=50, help="The number of coreset sizes to evaluate")
run_subparser.add_argument('--coreset_size_spacing', type=str, choices=['log', 'linear'], default='log', help="The spacing of coreset sizes to test")

# common arguments
run_subparser.add_argument('--trial', type=int, help='The trial number (used to seed random number generation)')
run_subparser.add_argument('--results_folder', type=str, default="results/", help="This script will save results in this folder")
run_subparser.add_argument('--verbosity', type=str, default="error", choices=['error', 'warning', 'critical', 'info', 'debug'], help="The verbosity level.")

# plotting arguments
plot_subparser.add_argument('plot_x', type = str, help="The X axis of the plot")
plot_subparser.add_argument('plot_y', type = str, help="The Y axis of the plot")
plot_subparser.add_argument('--plot_x_type', type=str, choices=["linear","log"], default = "log", help = "Specifies the scale for the X-axis")
plot_subparser.add_argument('--plot_y_type', type=str, choices=["linear","log"], default = "log", help = "Specifies the scale for the Y-axis.")
plot_subparser.add_argument('--plot_legend', type=str, help = "Specifies the variable to create a legend for.")
plot_subparser.add_argument('--plot_height', type=int, default=850, help = "Height of the plot's html canvas")
plot_subparser.add_argument('--plot_width', type=int, default=850, help = "Width of the plot's html canvas")
plot_subparser.add_argument('--plot_type', type=str, choices=['line', 'scatter'], default='scatter', help = "Type of plot to make")
plot_subparser.add_argument('--plot_fontsize', type=str, default='32pt', help = "Font size for the figure, e.g., 32pt")
plot_subparser.add_argument('--plot_toolbar', action='store_true', help = "Show the Bokeh toolbar")

arguments = parser.parse_args()
arguments.func(arguments)


