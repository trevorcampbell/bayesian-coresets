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

parser.add_argument('names', type = str, nargs = '+', default = ["SVI", "RAND", "GIGAO", "GIGAR"], help = "a list of which algorithm names to plot results for (Examples: SVI / GIGAO / GIGAR / RAND)")
trials = parser.add_mutually_exclusive_group(required=True)

trials.add_argument('--n_trials', type=int, help="Look for & plot experiments with trial IDs 1 through n_trials (inclusive)")
trials.add_argument('--seeds', type = int, nargs = '+', help="Plot experiments associated with the provided trial numbers (seeds)")

parser.add_argument('--fldr', type=str, default="results/", help="This script will look for & plot experiments in this folder")
parser.add_argument('--proj_dim', type=int, default = '100', help = "The number of samples taken when discretizing log likelihoods")
parser.add_argument('--SVI_opt_itrs', type=int, default = '2000', help = '(If using SVI/HOPS) The number of iterations used when optimizing weights.')
parser.add_argument('--SVI_step_sched', type=str, default = "lambda i : 1.e5/(1+i)", help="Step schedule (tuning rate) for SVI & HOPS, entered as a lambda expression in quotation marks. Default is \"lambda i : 1.e5/(1+i)\"")
parser.add_argument('--pihat_noise', type=float, default=.75, help = "(If calling GIGAR or simulating another realistically tuned Hilbert Coreset) - a measure of how much noise introduced to the smoothed pi-hat to make the sampler")
parser.add_argument('--n_subsample_opt', type=int, default=500, help="(If using Sparse VI/HOPS) the size of the random subsample used when optimizing the coreset weights in each reweight step")
parser.add_argument('--n_subsample_select', type=int, default=2000, help="(If using Sparse VI/HOPS) the size of the random subsample used when determining which point to add to the coreset in each select step")
parser.add_argument('--n_bases_per_scale', type=int, default=50, help="The number of Radial Basis Functions per scale")#TODO: verify help message


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
fldr = arguments.fldr

SVI_opt_itrs = arguments.SVI_opt_itrs
n_subsample_opt = arguments.n_subsample_opt
n_subsample_select = arguments.n_subsample_select
proj_dim = arguments.proj_dim
pihat_noise =arguments.pihat_noise
n_bases_per_scale = arguments.n_bases_per_scale
SVI_step_sched_hash_sha1_truncated = hashlib.sha1(arguments.SVI_step_sched.encode('utf-8')).hexdigest()[:10]

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
  kl = []
  sz = []
  for tr in trials:
    res = np.load((os.path.join(fldr, nm[0]+'_'+'tr='+str(tr)+'_n_bases_per_scale='+str(n_bases_per_scale)+'_proj_dim='+str(proj_dim)+
    '_SVI_opt_itrs='+str(SVI_opt_itrs)+'_n_sub_opt='+str(n_subsample_opt)+'_n_sub_sel='+str(n_subsample_select)+
    '_SVI_step_sched_hash_sha1_truncated='+SVI_step_sched_hash_sha1_truncated+'_pihat_noise='+str(pihat_noise)+'.npz')), 
    allow_pickle = True)

    data = { 'Iterations': [np.arange(1,len(res['rklw'])+1,plot_every)],
             'Coreset Size': [[np.count_nonzero(a) for a in res['w'][::plot_every]]],
             'Forward KL': [res['fklw'][::plot_every]],
             'Reverse KL': [res['rklw'][::plot_every]],
             'CPU Time(s)': [res['cputs'][::plot_every]]}
             
  x = np.percentile(data[X], 50, axis=0)
  fig.line(x, np.percentile(data[Y], 50, axis=0), color=pal[i-1], line_width=5, legend=nm[1])
  fig.patch(x = np.hstack((x, x[::-1])), y = np.hstack((np.percentile(data[Y], 75, axis=0), np.percentile(data[Y], 25, axis=0)[::-1])), color=pal[i-1], fill_alpha=0.4, legend=nm[1])

postprocess_plot(fig, '12pt', location='bottom_left', glyph_width=40)
fig.legend.background_fill_alpha=0.
fig.legend.border_line_alpha=0.
fig.legend.visible = True

bkp.show(fig)
