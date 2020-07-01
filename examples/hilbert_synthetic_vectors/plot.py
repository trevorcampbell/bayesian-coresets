import bokeh.plotting as bkp
import numpy as np
import sys, os
import argparse
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *

parser = argparse.ArgumentParser(description="Plots Riemannian linear regression experiments")
parser.add_argument('--X', type = str, default="Iterations", help="The X axis of the plot - one of Iterations/Coreset Size/Forward KL/Reverse KL/CPU Time(s)")
parser.add_argument('--X_scale', type=str, choices=["linear","log"], default = "log", help = "Specifies the scale for the X-axis. Default is \"log\".")
parser.add_argument('--Y', type = str, default = "Error", help="The Y axis of the plot - one of Iterations/Coreset Size/Forward KL/Reverse KL/CPU Time(s)")
parser.add_argument('--Y_scale', type=str, choices=["linear","log"], default = "log", help = "Specifies the scale for the Y-axis. Default is \"log\".")

parser.add_argument('--height', type=int, default=850, help = "Height of the plot's html canvas, default 850")
parser.add_argument('--width', type=int, default=850, help = "Width of the plot's html canvas, default 850")

parser.add_argument('alg_names', type=str, nargs = '+', choices=['FW', 'GIGA', 'OMP', 'IS', 'US'], help="The sparse non negative least squares algorithm(s) for which to plot results: one or more of FW (Frank Wolfe), GIGA (Greedy Iterative Geodeic Ascent), OMP (Orthogonal Matching Pursuit), IS (Importance Sampling), US (Uniform Sampling)")
trials = parser.add_mutually_exclusive_group(required=True)
trials.add_argument('--n_trials', type=int, help="Look for & plot experiments with trial IDs 1 through n_trials (inclusive)")
trials.add_argument('--seeds', type = int, nargs = '+', help="Plot experiments associated with the provided trial numbers (seeds)")

parser.add_argument('--fldr', type=str, default="results/", help="This script will look for & plot experiments in this folder")
parser.add_argument('--d', type=int, default = '100', help="The dimension of the datapoints used for these experiments (if the --diag flag is provided, this is also the number of synthetic data points)")
parser.add_argument('--N', type=int, default='10000', help='Dataset size/number of examples for these experiments (only if the --diag flag is not provided)')
parser.add_argument('--diag', action='store_const', default=False, const=True, help="If this flag is provided, plots results for an axis-aligned diagonal dataset (dxd) instead of the usual random Nxd matrix")


arguments = parser.parse_args()
X = arguments.X
X_scale = arguments.X_scale
Y = arguments.Y
Y_scale = arguments.Y_scale
height = arguments.height
width = arguments.width

names = arguments.alg_names
trials = np.arange(1, arguments.n_trials + 1) if arguments.n_trials else arguments.seeds
fldr = arguments.fldr
N = arguments.N
d = arguments.d
diag = arguments.diag

algs = {'FW': 'Frank Wolfe',
        'GIGA': 'GIGA', 
        'OMP': 'Orthogonal Matching Pursuit', 
        'IS': "Importance Sampling", 
        'US': "Uniform"
        }
nms = []
for name in names:
  nms.append((name, algs[name]))

#plot the KL figure
fig = bkp.figure(y_axis_type=Y_scale, x_axis_type=X_scale, plot_width=width, plot_height=height, x_axis_label=X, y_axis_label=Y, toolbar_location=None )
preprocess_plot(fig, '32pt', X_scale == 'log', Y_scale == 'log')

pal = bokeh.palettes.colorblind['Colorblind'][8]
pal = [pal[0], pal[1], '#d62728', pal[4], pal[6], pal[3], pal[7], pal[2]]
for i, nm in enumerate(nms):
  x_all = []
  y_all = []
  for tr in trials:
    numTuple = ("alg="+nm[0], "tr="+str(tr), "N="+str(N), "d="+str(d))
    print(os.path.join(fldr, 'gauss_results_'+('diag_' if diag else '')+'_'.join(numTuple)+'.npz'))
    res = np.load(os.path.join(fldr, 'gauss_results_'+('diag_' if diag else '')+'_'.join(numTuple)+'.npz'))
    data = { 'Iterations': res["Ms"],
             'Coreset Size': res['csize'],
             'Error': res['err'],
             'CPU Time(s)': res['cput']
             }
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
