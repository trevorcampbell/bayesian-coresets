import numpy as np
import scipy.linalg as sl
import pickle as pk
import os, sys
import argparse
import time
#make it so we can import models/etc from parent folder
import bayesiancoresets as bc
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import model_linreg 
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
    ## Step 0: Setup
    #######################################
    #######################################

    np.random.seed(arguments.trial)
    bc.util.set_verbosity(arguments.verbosity)
    
    if arguments.coreset_size_spacing == 'log':
        Ms = np.unique(np.logspace(0., np.log10(arguments.coreset_size_max), arguments.coreset_num_sizes, dtype=np.int32))
    else:
        Ms = np.unique(np.linspace(1, arguments.coreset_size_max, arguments.coreset_num_sizes, dtype=np.int32))

    #make sure the first size to record is 0
    if Ms[0] != 0:
        Ms = np.hstack((0, Ms))

    #######################################
    #######################################
    ## Step 1: Load and preprocess data
    #######################################
    #######################################
    
    #load data and compute true posterior
    #each row of x is [lat, lon, price]
    print('Loading data')
    
    x = np.load('../data/prices2018.npy')
    print('dataset size : ', x.shape)

    print('Subsampling down to '+str(arguments.data_num) + ' points')
    idcs = np.arange(x.shape[0])
    np.random.shuffle(idcs)
    x = x[idcs[:arguments.data_num], :]
    
    #log transform the prices
    x[:, 2] = np.log10(x[:, 2])
    
    #get empirical mean/std
    datastd = x[:,2].std()
    datamn = x[:,2].mean()
    
    #bases of increasing size; the last one is effectively a constant
    basis_unique_scales = np.array([.2, .4, .8, 1.2, 1.6, 2., 100])
    basis_unique_counts = np.hstack((arguments.n_bases_per_scale*np.ones(6, dtype=np.int64), 1))
    
    #the dimension of the scaling vector for the above bases
    d = basis_unique_counts.sum()
    print('Basis dimension: ' + str(d))
    
    #model params
    mu0 = datamn*np.ones(d)
    Sig0 = (datastd**2+datamn**2)*np.eye(d)
    Sig0inv = np.linalg.inv(Sig0)
    
    #generate basis functions by uniformly randomly picking locations in the dataset
    print('Trial ' + str(arguments.trial)) 
    print('Creating bases')
    basis_scales = np.array([])
    basis_locs = np.zeros((0,2))
    for i in range(basis_unique_scales.shape[0]):
      basis_scales = np.hstack((basis_scales, basis_unique_scales[i]*np.ones(basis_unique_counts[i])))
      idcs = np.random.choice(np.arange(x.shape[0]), replace=False, size=basis_unique_counts[i])
      basis_locs = np.vstack((basis_locs, x[idcs, :2]))
    
    print('Converting bases and observations into X/Y matrices')
    #convert basis functions + observed data locations into a big X matrix
    X = np.zeros((x.shape[0], basis_scales.shape[0]))
    for i in range(basis_scales.shape[0]):
      X[:, i] = np.exp( -((x[:, :2] - basis_locs[i, :])**2).sum(axis=1) / (2*basis_scales[i]**2) )
    Y = x[:, 2]
    Z = np.hstack((X, Y[:,np.newaxis]))

    _, bV = np.linalg.eigh(X.T.dot(X))
    bV = bV[:, -arguments.proj_dim:]

    #######################################
    #######################################
    ## Step 2: Calculate Likelihoods/Projectors
    #######################################
    #######################################
    
    #get true posterior
    print('Computing true posterior')
    mup, USigp, LSigpInv = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, Z, np.ones(X.shape[0]))
    Sigp = USigp.dot(USigp.T)
    SigpInv = LSigpInv.dot(LSigpInv.T)
    
    #create function to output log_likelihood given param samples
    print('Creating log-likelihood function')
    log_likelihood = lambda z, th : model_linreg.log_likelihood(z, th, datastd**2)
    
    print('Creating gradient log-likelihood function')
    grad_log_likelihood = lambda z, th : model_linreg.grad_x_log_likelihood(z, th, datastd**2)
    
    #create tangent space for well-tuned Hilbert coreset alg
    print('Creating tuned projector for Hilbert coreset construction')
    sampler_optimal = lambda n, w, pts : mup + np.random.randn(n, mup.shape[0]).dot(USigp.T)
    prj_optimal = bc.BlackBoxProjector(sampler_optimal, arguments.proj_dim, log_likelihood, grad_log_likelihood)
    
    #create tangent space for poorly-tuned Hilbert coreset alg
    print('Creating untuned projector for Hilbert coreset construction')
    Zhat = Z[np.random.randint(0, Z.shape[0], int(np.sqrt(Z.shape[0]))), :]
    muhat, USigHat, LSigHatInv = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, Zhat, np.ones(Zhat.shape[0]))
    sampler_realistic = lambda n, w, pts : muhat + np.random.randn(n, muhat.shape[0]).dot(USigHat.T)
    prj_realistic = bc.BlackBoxProjector(sampler_realistic, arguments.proj_dim, log_likelihood, grad_log_likelihood)

    print('Creating black box projector')
    def sampler_w(n, wts, pts):
        if wts is None or pts is None or pts.shape[0] == 0:
            muw = mu0
            USigw = np.linalg.cholesky(Sig0) #Note: USigw is lower triangular here, below is upper tri. Doesn't matter, just need Sigw = MM^T
        else:
            muw, USigw, _ = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, pts, wts)
        return muw + np.random.randn(n, muw.shape[0]).dot(USigw.T)
    prj_bb = bc.BlackBoxProjector(sampler_w, arguments.proj_dim, log_likelihood, grad_log_likelihood)

    print('Creating exact projectors')
    ##############################
    ###Exact projection in SparseVI for gradient computation
    #for this model we can do the tangent space projection exactly
    class LinRegProjector(bc.Projector):
        def __init__(self, bV):
            self.bV = bV

        def project(self, pts, grad=False):
            X = pts[:, :-1]
            Y = pts[:, -1]
            #beta = X.dot(self.V*np.sqrt(np.maximum(self.lmb, 0.)))
            beta = X.dot(self.USigw)
            nu = Y - X.dot(self.muw)
            #approximation to avoid high memory cost: project the matrix term down to bV.shape[1]**2 dimensions
            beta_proj = beta.dot(self.bV)
            #lmb2, V2 = np.linalg.eigh(beta.T.dot(beta))
            #beta_proj = beta.dot(V2[:, -arguments.proj_dim:])
            return np.hstack((nu[:, np.newaxis]*beta, 1./np.sqrt(2.)*(beta_proj[:, :, np.newaxis]*beta_proj[:, np.newaxis, :]).reshape(beta.shape[0], arguments.proj_dim**2))) / datastd**2
    
        def update(self, wts, pts):
            if wts is None or pts is None or pts.shape[0] == 0:
                self.muw = mu0
                self.USigw = np.linalg.cholesky(Sig0) #Note: USigw here is lower triangular, but keeping naming convention for below stuff. Doesn't matter, just need Sigw = MM^T
            else:
                self.muw, self.USigw, _ = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, pts, wts)
            #if pts.shape[0] == 0:
            #    self.muw = mu0
            #    self.Sigw = Sig0
            #else:
            #    self.muw, self.Sigw = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, pts, wts)
            #self.lmb, self.V = np.linalg.eigh(self.LSigw.dot(self.LSigw.T))

    prj_optimal_exact = LinRegProjector(bV)
    prj_optimal_exact.update(np.ones(Z.shape[0]), Z)
    prj_realistic_exact = LinRegProjector(bV)
    prj_realistic_exact.update(np.ones(Zhat.shape[0]), Zhat)
    
    #######################################
    #######################################
    ## Step 3: Construct Coreset
    #######################################
    #######################################
    
    ##############################
    print('Creating coreset construction objects')
    #create coreset construction objects
    sparsevi_exact = bc.SparseVICoreset(Z, LinRegProjector(bV), opt_itrs = arguments.opt_itrs, step_sched = eval(arguments.step_sched))
    sparsevi = bc.SparseVICoreset(Z, prj_bb, opt_itrs = arguments.opt_itrs, step_sched = eval(arguments.step_sched))
    giga_optimal = bc.HilbertCoreset(Z, prj_optimal)
    giga_optimal_exact = bc.HilbertCoreset(Z,prj_optimal_exact)
    giga_realistic = bc.HilbertCoreset(Z,prj_realistic)
    giga_realistic_exact = bc.HilbertCoreset(Z,prj_realistic_exact)
    unif = bc.UniformSamplingCoreset(Z)
    
    algs = {'SVI-EXACT': sparsevi_exact,
            'SVI': sparsevi, 
            'GIGA-OPT': giga_optimal, 
            'GIGA-OPT-EXACT': giga_optimal_exact, 
            'GIGA-REAL': giga_realistic, 
            'GIGA-REAL-EXACT': giga_realistic_exact, 
            'US': unif}
    alg = algs[arguments.alg]
    
    print('Building coreset')
    w = []
    p = []
    cputs = np.zeros(Ms.shape[0])
    t_build = 0
    for m in range(Ms.shape[0]):
      print('M = ' + str(Ms[m]) + ': coreset construction, '+ arguments.alg + ' ' + str(arguments.trial))
      t0 = time.process_time()
      itrs = (Ms[m] if m == 0 else Ms[m] - Ms[m-1])
      alg.build(itrs)
      t_build += time.process_time()-t0
      wts, pts, idcs = alg.get()
    
      #store weights/pts/runtime
      w.append(wts)
      p.append(pts)
      cputs[m] = t_build
    
    ##############################
    ##############################
    ## Step 4: Evaluate coreset
    ##############################
    ##############################
    
    # computing kld and saving results
    muw = np.zeros((Ms.shape[0], mu0.shape[0]))
    Sigw = np.zeros((Ms.shape[0], mu0.shape[0], mu0.shape[0]))
    rklw = np.zeros(Ms.shape[0])
    fklw = np.zeros(Ms.shape[0])
    mu_errs = np.zeros(Ms.shape[0])
    Sig_errs = np.zeros(Ms.shape[0])
    csizes = np.zeros(Ms.shape[0])
    for m in range(Ms.shape[0]):
      csizes[m] = (w[m] > 0).sum()
      muw[m, :], USigw, LSigwInv = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, p[m], w[m])
      Sigw[m, :, :] = USigw.dot(USigw.T)
      rklw[m] = model_linreg.KL(muw[m,:], Sigw[m,:,:], mup, SigpInv)
      fklw[m] = model_linreg.KL(mup, Sigp, muw[m,:], LSigwInv.dot(LSigwInv.T))
      mu_errs[m] = np.sqrt(((mup - muw[m,:])**2).sum()) / np.sqrt((mup**2).sum())
      Sig_errs[m] = np.sqrt(((Sigp - Sigw[m,:,:])**2).sum()) / np.sqrt((Sigp**2).sum())

    results.save(arguments, csizes = csizes, Ms = Ms, cputs = cputs, rklw = rklw, fklw = fklw, mu_errs = mu_errs, Sig_errs = Sig_errs)

    #also save muw/Sigw/etc for plotting coreset visualizations
    f = open('results/coreset_data.pk', 'wb')
    res = (x, mu0, Sig0, datastd, mup, Sigp, w, p, muw, Sigw)
    pk.dump(res, f)
    f.close()

############################
############################
## Parse arguments
############################
############################
 
parser = argparse.ArgumentParser(description="Runs Riemannian linear regression (employing coreset contruction) on the specified dataset")
subparsers = parser.add_subparsers(help='sub-command help')
run_subparser = subparsers.add_parser('run', help='Runs the main computational code')
run_subparser.set_defaults(func=run)
plot_subparser = subparsers.add_parser('plot', help='Plots the results')
plot_subparser.set_defaults(func=plot)

parser.add_argument('--data_num', type=int, default='10000', help='Dataset subsample to use')
parser.add_argument('--alg', type=str, default='SVI', choices = ['SVI', 'SVI-EXACT', 'GIGA-OPT', 'GIGA-OPT-EXACT', 'GIGA-REAL', 'GIGA-REAL-EXACT', 'US'], help="The name of the coreset construction algorithm to use")
parser.add_argument("--proj_dim", type=int, default=100, help="The number of samples taken when discretizing log likelihoods for these experiments")

parser.add_argument('--coreset_size_max', type=int, default=300, help="The maximum coreset size to evaluate")
parser.add_argument('--coreset_num_sizes', type=int, default=6, help="The number of coreset sizes to evaluate")
parser.add_argument('--coreset_size_spacing', type=str, choices=['log', 'linear'], default='log', help="The spacing of coreset sizes to test")

parser.add_argument('--n_bases_per_scale', type=int, default=50, help="The number of Radial Basis Functions per scale")#TODO: verify help message
parser.add_argument('--opt_itrs', type=str, default = 100, help="Number of optimization iterations (for methods that use iterative weight refinement)")
parser.add_argument('--step_sched', type=str, default = "lambda i : 1./(1+i)", help="Optimization step schedule (for methods that use iterative weight refinement); entered as a python lambda expression surrounded by quotes")

parser.add_argument('--trial', type=int, help="The trial number - used to initialize random number generation (for replicability)")
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


