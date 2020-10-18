from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize
import time
import sys, os
import argparse
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import mcmc
import results
import plotting

def plot(arguments):
    # load only the results that match (avoid high mem usage)
    to_match = vars(arguments)
    #remove any ignored params
    for nm in arguments.summarize:
        to_match.pop(nm, None)
    #remove any legend param
    to_match.pop(arguments.plot_legend, None)
    #load cols from results dfs that match remaining keys
    resdf = results.load_matching(to_match)
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
    ## Step 1: Define Model
    #######################################
    #######################################
    
    if arguments.model=="lr":
      import model_lr as model
    elif arguments.model=="poiss":
      import model_poiss as model
    
    #######################################
    #######################################
    ## Step 2: Load Dataset
    #######################################
    #######################################
    
    print('Loading dataset '+arguments.dataset)
    X,Y,Z, Zt, D = model.load_data('../data/'+arguments.dataset+'.npz')
    
    #############################################################################
    #############################################################################
    ## Step 3: Make a Laplace Approximation to Inform the Coreset Tangent Space
    #############################################################################
    #############################################################################
    
    
    if not os.path.exists('laplace_cache/'):
      os.mkdir('laplace_cache')  
    
    if not os.path.exists('laplace_cache/'+arguments.dataset+'_laplace.npz'):
      print('Computing Laplace approximation for '+arguments.dataset)
      t0 = time.process_time()
      res = minimize(lambda mu : -model.log_joint(Z, mu, np.ones(Z.shape[0]))[0], Z.mean(axis=0)[:D], jac=lambda mu : -model.grad_th_log_joint(Z, mu, np.ones(Z.shape[0]))[0,:])
      mu = res.x
      cov = -np.linalg.inv(model.hess_th_log_joint(Z, mu, np.ones(Z.shape[0]))[0,:,:])
      t_laplace = time.process_time() - t0
      np.savez('laplace_cache/'+arguments.dataset+'_laplace.npz', mu=mu, cov=cov, t_laplace=t_laplace)
    else:
      print('Loading Laplace approximation for '+arguments.dataset)
      lplc = np.load('laplace_cache/'+arguments.dataset+'_laplace.npz')
      mu = lplc['mu']
      cov = lplc['cov']
      t_laplace = lplc['t_laplace']
    
    ##########################################################################
    ##########################################################################
    ## Step 3: Compute a random finite projection of the tangent space  
    ##########################################################################
    ##########################################################################
    
    #generate a sampler based on the laplace approx 
    sampler = lambda sz, w, pts : np.atleast_2d(np.random.multivariate_normal(mu, cov, sz))
    projector = bc.BlackBoxProjector(sampler, arguments.proj_dim, model.log_likelihood)
    
    #########################################################################
    #########################################################################
    ## Step 4: Run MCMC on full dataset (important for coreset evaluation)
    #########################################################################
    #########################################################################
    
    full_samples = mcmc.sampler(arguments.dataset, X, Y, arguments.mcmc_samples_full, arguments.model, model.stan_representation, sample_caching_folder = "mcmc_cache/")
    #adjusting the format of samples returned by stan to match our expected format (see https://github.com/trevorcampbell/bayesian-coresets-private/issues/57)
    full_samples = np.hstack((full_samples[:, 1:], full_samples[:, 0][:,np.newaxis]))
    
    ######################################
    ######################################
    ## Step 5: Build/Evaluate the Coreset
    ######################################
    ######################################
    
    cputs = np.zeros(Ms.shape[0])
    mcmc_time_per_itr = np.zeros(Ms.shape[0])
    csizes = np.zeros(Ms.shape[0])
    Fs = np.zeros(Ms.shape[0])
    
    print('Running coreset construction / MCMC for ' + arguments.dataset + ' ' + arguments.alg + ' ' + str(arguments.trial))
    t0 = time.process_time()
    alg = bc.HilbertCoreset(Z, projector, snnls = algs[arguments.alg])
    t_setup = time.process_time() - t0
    t_alg = 0.
    for m in range(Ms.shape[0]):
      print('M = ' + str(Ms[m]) + ': coreset construction, '+ arguments.alg + ' ' + arguments.dataset + ' ' + str(arguments.trial))
      #this runs alg up to a level of M; on the next iteration, it will continue from where it left off
      t0 = time.process_time()
      itrs = (Ms[m] if m == 0 else Ms[m] - Ms[m-1])
      alg.build(itrs)
      t_alg += time.process_time()-t0
      wts, pts, idcs = alg.get()
    
    
      print('M = ' + str(Ms[m]) + ': MCMC')
      # Here we would like to measure the time it would take to run mcmc on our coreset.
      # however, this is particularly challenging - stan's mcmc implementation doesn't work on 
      # the weighted likelihoods we use in our coresets. And our inference.py nuts implementation
      # (which does work on weighted likelihoods) is not as efficient or reliable as stan.
      curX = X[idcs, :]
      curY = Y[idcs]
      t0 = time.process_time()
      mcmc.sampler(arguments.dataset, curX, curY, arguments.mcmc_samples_coreset, arguments.model, model.stan_representation, weights=wts)
      t_alg_mcmc = time.process_time()-t0 
      t_alg_mcmc_per_iter = t_alg_mcmc/(arguments.mcmc_samples_coreset*2) #if we change the number of burn_in steps to differ from the number of actual samples we take, we might need to reconsider this line  
    
      print('M = ' + str(Ms[m]) + ': CPU times')
      cputs[m] = t_laplace + t_setup + t_alg
      mcmc_time_per_itr[m] = t_alg_mcmc_per_iter
      print('M = ' + str(Ms[m]) + ': coreset sizes')
      csizes[m] = wts.shape[0]
      print('M = ' + str(Ms[m]) + ': F norms')
      gcs = np.array([ model.grad_th_log_joint(Z[idcs, :], full_samples[i, :], wts) for i in range(full_samples.shape[0]) ])
      gfs = np.array([ model.grad_th_log_joint(Z, full_samples[i, :], np.ones(Z.shape[0])) for i in range(full_samples.shape[0]) ])
      Fs[m] = (((gcs - gfs)**2).sum(axis=1)).mean()
    
    results.save(arguments, csizes = csizes, Ms = Ms, cputs = cputs, Fs = Fs, mcmc_time_per_itr = mcmc_time_per_itr)
    #np.savez_compressed('results/'+arguments.dataset+'_'+model+'_'+arguments.alg+'_results_'+'id='+str(ID)+"_mcmc_samples_coreset="+str(mcmc_samples_coreset)+"_mcmc_samples_full="+str(mcmc_samples_full) + "_proj_dim="+str(projection_dim)+'_Ms='+str(Ms)+'.npz', Ms=Ms, Fs=Fs, cputs=cputs, mcmc_time_per_itr = mcmc_time_per_itr, csizes=csizes, mcmc_samples_coreset=mcmc_samples_coreset, mcmc_samples_full=mcmc_samples_full, proj_dim=projection_dim)
    


############################
############################
## Parse arguments
############################
############################
 
parser = argparse.ArgumentParser(description="Runs Hilbert coreset construction on a model and dataset")
subparsers = parser.add_subparsers(help='sub-command help')
run_subparser = subparsers.add_parser('run', help='Runs the main computational code')
run_subparser.set_defaults(func=run)
plot_subparser = subparsers.add_parser('plot', help='Plots the results')
plot_subparser.set_defaults(func=plot)


parser.add_argument('--model', type=str, choices=["lr","poiss"], help="The model to use.") #must be one of linear regression or poisson regression
parser.add_argument('--dataset', type=str, help="The name of the dataset") #examples: synth_lr, phishing, ds1, synth_poiss, biketrips, airportdelays, synth_poiss_large, biketrips_large, airportdelays_large
parser.add_argument('--alg', type=str, default='GIGA', choices = ['GIGA', 'FW', 'US', 'OMP'], help="The algorithm to use for solving sparse non-negative least squares - should be one of GIGA / FW / US / OMP") #TODO: find way to make this help message autoupdate with new methods
parser.add_argument("--mcmc_samples_full", type=int, default=10000, help="number of MCMC samples to take for inference on the full dataset (also take this many warmup steps before sampling)")
parser.add_argument("--mcmc_samples_coreset", type=int, default=10000, help="number of MCMC samples to take for inference on the coreset (also take this many warmup steps before sampling)")
parser.add_argument("--proj_dim", type=int, default=500, help="The number of samples taken when discretizing log likelihoods for these experiments")

parser.add_argument('--coreset_size_max', type=int, default=1000, help="The maximum coreset size to evaluate")
parser.add_argument('--coreset_num_sizes', type=int, default=7, help="The number of coreset sizes to evaluate")
parser.add_argument('--coreset_size_spacing', type=str, choices=['log', 'linear'], default='log', help="The spacing of coreset sizes to test")

parser.add_argument('--trial', type=int, help="The trial number - used to initialize random number generation (for replicability)")
parser.add_argument('--results_folder', type=str, default="results/", help="This script will save results in this folder")
parser.add_argument('--verbosity', type=str, default="error", choices=['error', 'warning', 'critical', 'info', 'debug'], help="The verbosity level.")

# plotting arguments
plot_subparser.add_argument('plot_x', type = str, help="The X axis of the plot")
plot_subparser.add_argument('plot_y', type = str, help="The Y axis of the plot")
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


