import bokeh.plotting as bkp
from bokeh.io import export_png, export_svgs
import numpy as np
import sys, os
import argparse
import hashlib
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *

parser = argparse.ArgumentParser(description="Plots Riemannian linear regression experiments")
parser.add_argument('--X', type = str, default="Iterations", help="The X axis of the plot - one of Iterations/Coreset Size/Forward KL/Reverse KL/CPU Time(s)")
parser.add_argument('--X_scale', type=str, choices=["linear","log"], default = "linear", help = "Specifies the scale for the X-axis. Default is \"linear\".")
parser.add_argument('--Y', type = str, default = "Reverse KL", help="The Y axis of the plot - one of Iterations/Coreset Size/Forward KL/Reverse KL/CPU Time(s)")
parser.add_argument('--Y_scale', type=str, choices=["linear","log"], default = "log", help = "Specifies the scale for the Y-axis. Default is \"log\".")

parser.add_argument('--height', type=int, default=850, help = "Height of the plot's html canvas, default 850")
parser.add_argument('--width', type=int, default=850, help = "Width of the plot's html canvas, default 850")
parser.add_argument('--plot_every', type=int, default='1', help="Coarseness of the graph - will skip (plot_every-1) points between each plotted point")

parser.add_argument('dnm', type=str, help="the name of the dataset for which to plot results")
parser.add_argument('model', type=str, choices=["lr","poiss"], help="The regression model used. lr refers to logistic regression, and poiss refers to poisson regression.")

parser.add_argument('names', type = str, nargs = '+', default = ["SVI", "RAND", "GIGAO", "GIGAR"], help = "a list of which algorithm names to plot results for (Examples: SVI / GIGAO / GIGAR / RAND)")
trials = parser.add_mutually_exclusive_group(required=True)

trials.add_argument('--n_trials', type=int, help="Look for & plot experiments with trial IDs 1 through n_trials (inclusive)")
trials.add_argument('--seeds', type = int, nargs = '+', help="Plot experiments associated with the provided trial numbers (seeds)")

parser.add_argument('--fldr', type=str, default="results/", help="This script will look for & plot experiments in this folder")
parser.add_argument('--proj_dim', type=int, default = '100', help = "The number of samples taken when discretizing log likelihoods for these experiments")
parser.add_argument('--SVI_opt_itrs', type=int, default = '1500', help = '(If using SVI/HOPS) The number of iterations used when optimizing weights.')
parser.add_argument('--SVI_step_sched', type=str, default = "lambda i : 1./(1+i)", help="Plots code with the associated step schedule (tuning rate) for SVI & HOPS. Default is \"lambda i : 1./(1+i)\", with the quotation marks.")
parser.add_argument('--pihat_noise', type=float, default=.75, help = "(If plotting GIGAR or simulating another realistically tuned Hilbert Coreset) - plots data corresponding to this much noise being introduced to the smoothed pi-hat to make the sampler")

parser.add_argument('--use_diag_laplace_w', action='store_const', default = False, const=True, help="")
parser.add_argument('--n_subsample_opt', type=int, default=400, help="(If using Sparse VI/HOPS) the size of the random subsample used when optimizing the coreset weights in each reweight step")
parser.add_argument('--n_subsample_select', type=int, default=1000, help="(If using Sparse VI/HOPS) the size of the random subsample used when determining which point to add to the coreset in each select step")
parser.add_argument("--mcmc_samples", type=int, default=10000, help="number of MCMC samples taken (we also took this many warmup steps before sampling)")

arguments = parser.parse_args()
X = arguments.X
X_scale = arguments.X_scale
Y = arguments.Y
Y_scale = arguments.Y_scale
height = arguments.height
width = arguments.width
plot_every = arguments.plot_every

names = arguments.names
trials = np.arange(1, arguments.n_trials + 1) if arguments.n_trials else arguments.seeds
model = arguments.model
dnm = arguments.dnm
fldr = arguments.fldr

proj_dim = arguments.proj_dim
SVI_opt_itrs =  arguments.SVI_opt_itrs
SVI_step_sched_hash_sha1 = hashlib.sha1(arguments.SVI_step_sched.encode('utf-8')).hexdigest()
pihat_noise = arguments.pihat_noise
use_diag_laplace_w = arguments.use_diag_laplace_w
n_subsample_opt = arguments.n_subsample_opt
n_subsample_select = arguments.n_subsample_select
N_samples = arguments.mcmc_samples

algs = {'SVIEXACT': 'Sparse VI (Exact Tangent Space)',
        'SVI': 'Sparse VI', 
        'GIGAO': 'GIGA(Optimal)', 
        'GIGAR': "GIGA(Realistic)", 
        'RAND': "Uniform"}
nms = []
for name in names:
  nms.append((name, algs[name]))

#plot the KL figure
fig = bkp.figure(y_axis_type=Y_scale, x_axis_type=X_scale, plot_width=width, plot_height=height, x_axis_label=X, y_axis_label=Y, toolbar_location=None )
preprocess_plot(fig, '32pt', X_scale == 'log', Y_scale == 'log')

for i, nm in enumerate(nms):
  x_all = []
  y_all = []
  for tr in trials:
    numTuple = (dnm, model, nm[0], "results", "id="+str(tr), "mcmc_samples="+str(N_samples), "use_diag_laplace_w="+str(use_diag_laplace_w), "proj_dim="+str(proj_dim), "SVI_opt_itrs="+str(SVI_opt_itrs), 'n_subsample_opt='+str(n_subsample_opt), "n_subsample_select="+str(n_subsample_select), 'SVI_step_sched_hash_sha1='+str(SVI_step_sched_hash_sha1), 'pihat_noise='+str(pihat_noise))
    print(os.path.join(fldr, '_'.join(numTuple)+'.npz'))
    res = np.load(os.path.join(fldr, '_'.join(numTuple)+'.npz'), allow_pickle = True)
    data = { 'Iterations': np.arange(1,len(res['rkls_laplace'])+1,plot_every),
             'Coreset Size': [np.count_nonzero(a) for a in res['w'][::plot_every]],
             'Forward KL': res['fkls_laplace'][::plot_every],
             'Reverse KL': res['rkls_laplace'][::plot_every],
             'CPU Time(s)': res['cputs'][::plot_every]}
    x_all.append(data[X])
    y_all.append(data[Y])

  x = np.percentile(x_all, 50, axis=0)
  fig.line(x, np.percentile(y_all, 50, axis=0), color=pal[i-1], line_width=5, legend=nm[1])
  fig.patch(x = np.hstack((x, x[::-1])), y = np.hstack((np.percentile(y_all, 75, axis=0), np.percentile(y_all, 25, axis=0)[::-1])), color=pal[i-1], fill_alpha=0.4, legend=nm[1])

postprocess_plot(fig, '12pt', location='bottom_left', glyph_width=40)
fig.legend.background_fill_alpha=0.
fig.legend.border_line_alpha=0.
fig.legend.visible = True

bkp.show(fig)
