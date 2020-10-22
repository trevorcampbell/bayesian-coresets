from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
import time
import os
import sys
import argparse
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import results
import plotting

def plot(arguments):
    # load only the results that match (avoid high mem usage)
    to_match = vars(arguments)
    #remove any ignored params
    if arguments.summarize is not None:
        for nm in arguments.summarize:
            to_match.pop(nm, None)
    #remove any legend param
    to_match.pop(arguments.plot_legend, None)
    #load cols from results dfs that match remaining keys
    resdf = results.load_matching(to_match)

    if resdf is None:
        print('No matching results to plot, skipping')
        return

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
    
    print('data: ' + arguments.data_type + ', trial ' + str(arguments.trial) + ', alg: ' + arguments.alg)

    class IDProjector(bc.Projector):
      def update(self, wts, pts):
        pass

      def project(self, pts, grad=False):
        return pts

    alg = bc.HilbertCoreset(X, IDProjector(), snnls = algs[arguments.alg])
    
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
 

parser = argparse.ArgumentParser('Runs sparse nonnegative regression')
subparsers = parser.add_subparsers(help='sub-command help')
run_subparser = subparsers.add_parser('run', help='Runs the main computational code')
run_subparser.set_defaults(func=run)
plot_subparser = subparsers.add_parser('plot', help='Plots the results')
plot_subparser.set_defaults(func=plot)

# example-specific arguments
parser.add_argument('--alg', type=str, default='GIGA', choices=['FW', 'GIGA', 'OMP', 'US'], help="The sparse non negative least squares algorithm to use")
parser.add_argument('--data_num', type=int, default=10000, help="The number of synthetic data points")
parser.add_argument('--data_dim', type=int, default=100, help="The dimension of the synthetic data points, if applicable")
parser.add_argument('--data_type', type=str, default='normal', choices=['normal', 'axis'], help="Specifies the type of synthetic data to generate.")
parser.add_argument('--coreset_size_max', type=int, default=1000, help="The maximum coreset size to evaluate")
parser.add_argument('--coreset_num_sizes', type=int, default=50, help="The number of coreset sizes to evaluate")
parser.add_argument('--coreset_size_spacing', type=str, choices=['log', 'linear'], default='log', help="The spacing of coreset sizes to test")

# common arguments
parser.add_argument('--trial', type=int, help='The trial number (used to seed random number generation)')
parser.add_argument('--results_folder', type=str, default="results/", help="This script will save results in this folder")
parser.add_argument('--verbosity', type=str, default="error", choices=['error', 'warning', 'critical', 'info', 'debug'], help="The verbosity level.")

# plotting arguments
plot_subparser.add_argument('plot_x', type = str, help="The X axis of the plot")
plot_subparser.add_argument('plot_y', type = str, help="The Y axis of the plot")
plot_subparser.add_argument('--plot_title', type = str, help="The title of the plot")
plot_subparser.add_argument('--plot_x_label', type = str, help="The X axis label of the plot")
plot_subparser.add_argument('--plot_y_label', type = str, help="The Y axis label of the plot")
plot_subparser.add_argument('--plot_x_type', type=str, choices=["linear","log"], default = "log", help = "Specifies the scale for the X-axis")
plot_subparser.add_argument('--plot_y_type', type=str, choices=["linear","log"], default = "log", help = "Specifies the scale for the Y-axis.")
plot_subparser.add_argument('--plot_legend', type=str, help = "Specifies the variable to create a legend for.")
plot_subparser.add_argument('--plot_height', type=int, default=850, help = "Height of the plot's html canvas")
plot_subparser.add_argument('--plot_width', type=int, default=850, help = "Width of the plot's html canvas")
plot_subparser.add_argument('--plot_type', type=str, choices=['line', 'scatter'], default='scatter', help = "Type of plot to make")
plot_subparser.add_argument('--plot_fontsize', type=str, default='32pt', help = "Font size for the figure, e.g., 32pt")
plot_subparser.add_argument('--plot_toolbar', action='store_true', help = "Show the Bokeh toolbar")
plot_subparser.add_argument('--summarize', type=str, nargs='*', help = 'The command line arguments to ignore value of when matching to plot a subset of data. E.g. --summarize trial data_num will compute result statistics over both trial and number of datapoints')
plot_subparser.add_argument('--groupby', type=str, help = 'The command line argument group rows by before plotting. No groupby means plotting raw data; groupby will do percentile stats for all data with the same groupby value. E.g. --groupby Ms in a scatter plot will compute result statistics for fixed values of M, i.e., there will be one scatter point per value of M')

arguments = parser.parse_args()
arguments.func(arguments)


