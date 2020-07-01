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
parser.add_argument('--X', type = str, default="Iterations", help="The X axis of the plot - one of Iterations/Coreset Size/F Norms/CPU Time(s)")
parser.add_argument('--X_scale', type=str, choices=["linear","log"], default = "linear", help = "Specifies the scale for the X-axis. Default is \"linear\".")
parser.add_argument('--Y', type = str, default = "F Norms", help="The Y axis of the plot - one of Iterations/Coreset Size/F Norms/CPU Time(s)")
parser.add_argument('--Y_scale', type=str, choices=["linear","log"], default = "log", help = "Specifies the scale for the Y-axis. Default is \"log\".")

parser.add_argument('--height', type=int, default=850, help = "Height of the plot's html canvas, default 850")
parser.add_argument('--width', type=int, default=850, help = "Width of the plot's html canvas, default 850")

parser.add_argument('model', type=str, choices=["lr","poiss"], help="The regression model used. lr refers to logistic regression, and poiss refers to poisson regression.")
parser.add_argument('dnm', type=str, help="the name of the dataset for which to plot results")

parser.add_argument('names', type = str, nargs = '+', default = ["FW", "RND", "GIGA"], help = "a list of which algorithm names to plot results for")
trials = parser.add_mutually_exclusive_group(required=True)

trials.add_argument('--n_trials', type=int, help="Look for & plot experiments with trial IDs 1 through n_trials (inclusive)")
trials.add_argument('--seeds', type = int, nargs = '+', help="Plot experiments associated with the provided trial numbers (seeds)")

parser.add_argument('--fldr', type=str, default="results/", help="This script will look for & plot experiments in this folder")
parser.add_argument('--proj_dim', type=int, default = 500, help = "The number of samples taken when discretizing log likelihoods for these experiments")

parser.add_argument("--mcmc_samples_full", type=int, default=10000, help="number of MCMC samples to take for the posterior estimate using the full dataset (we also take this many warmup steps before sampling)")
parser.add_argument("--mcmc_samples_coreset", type=int, default=10000, help="number of MCMC samples to take for the posterior estimate using our coreset (we also take this many warmup steps before sampling)")
parser.add_argument("--Ms", type=int, nargs='+', default = None, help = "A list of M values (maximum allowable coreset sizes) to try in this experiment.")
parser.add_argument("--num_Ms", type=int, default=10, help="(if --Ms is not specified) The number of different coreset sizes (Ms) to try in this experiment (We try Ms from 1 to --M_max, inclusive, with logarithmic spacing between values of M). Default 10.")
parser.add_argument("--M_max", type=int, default=1000, help="(if --Ms is not specified) The largest coreset size to try in this experiment. Default 1000.")

arguments = parser.parse_args()
X = arguments.X
X_scale = arguments.X_scale
Y = arguments.Y
Y_scale = arguments.Y_scale
height = arguments.height
width = arguments.width

names = arguments.names
trials = np.arange(1, arguments.n_trials + 1) if arguments.n_trials else arguments.seeds
model = arguments.model
dnm = arguments.dnm
fldr = arguments.fldr

mcmc_samples_full = arguments.mcmc_samples_full
mcmc_samples_coreset = arguments.mcmc_samples_coreset
proj_dim = arguments.proj_dim

Ms = arguments.Ms if arguments.Ms is not None else np.unique(np.logspace(0, np.log10(arguments.M_max), arguments.num_Ms, dtype=int))

algs = {'FW': 'Frank-Wolfe',
        'GIGA': "GIGA", 
        'RND': "Uniform"}
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
    numTuple = (dnm, model, nm[0], "results", "id="+str(tr), "mcmc_samples_coreset="+str(mcmc_samples_coreset), "mcmc_samples_full="+str(mcmc_samples_full), "proj_dim="+str(proj_dim), 'Ms='+str(Ms))
    print(os.path.join(fldr, '_'.join(numTuple)+'.npz'))
    res = np.load(os.path.join(fldr, '_'.join(numTuple)+'.npz'), allow_pickle = True)
    data = { 'Iterations': res['Ms'],
             'Coreset Size': res['csizes'],
             'F Norms': res['Fs'],
             'CPU Time(s)': res['cputs']}
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
