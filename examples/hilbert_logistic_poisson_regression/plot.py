import numpy as np
import bokeh.plotting as bkp
import bokeh.io as bki
import bokeh.layouts as bkl
from bokeh.models import FuncTickFormatter
import bokeh.models as bkm
import bokeh.palettes 
import time
import os, sys
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *

dnames = ['synth_lr', 'biketrips', 'airportdelays', 'synth_poiss', 'ds1', 'phishing']
algs = [('RND', 'Uniform', pal[7]),  ('FW', 'Frank Wolfe', pal[2]),('GIGA','GIGA', pal[1])]

fig_cput = bkp.figure(y_axis_type='log', y_axis_label='Normalized Fisher Information Distance', x_axis_type='log', x_axis_label='Relative Total CPU Time', x_range=(.05, 1.1), plot_width=1250, plot_height=1250)
fig_csz = bkp.figure(y_axis_type='log', y_axis_label='Normalized Fisher Information Distance', x_axis_type='log', x_axis_label='Coreset Size', plot_width=1250, plot_height=1250)

preprocess_plot(fig_cput, '32pt', False, True)
preprocess_plot(fig_csz, '32pt', True, True)

dnmsalgs = [(dnm, alg) for dnm in dnames for alg in algs]

#get RND median normalization
NMs = -1
std_Fs = {}
std_ts = {}
for didx, dnm in enumerate(dnames):
  trials = [fn for fn in os.listdir('results/') if dnm+'_RND_results_' in fn]
  if len(trials) == 0: 
    print('Need to run RND to establish baseline first')
    quit()
  Fs = []
  for tridx, fn in enumerate(trials):
    res = np.load('results/'+fn)
    Fs.append(res['Fs'])
    NMs = res['Ms'].shape[0]
  std_Fs[dnm] = np.percentile(np.array(Fs), 50)
  std_ts[dnm] = np.load('results/'+dnm+'_posterior_samples.npz')['t_full']

for idx, zppd in enumerate(dnmsalgs):
  dnm, alg = zppd
  trials = [fn for fn in os.listdir('results/') if dnm+'_'+alg[0]+'_results_' in fn]
  if len(trials) == 0: continue
  cputs = np.zeros((len(trials), NMs))
  cszs = np.zeros((len(trials), NMs))
  Fs = np.zeros((len(trials), NMs))
  for tridx, fn in enumerate(trials):
    res = np.load('results/'+fn)
    cputs[tridx, :] = res['cputs']
    Fs[tridx, :] = res['Fs']
    cszs[tridx, :] = res['csizes']

  fig_cput.line(np.percentile(cputs, 50, axis=0)/std_ts[dnm], np.percentile(Fs, 50, axis=0)/std_Fs[dnm], line_color=alg[2], line_width=8, legend=alg[1])
  fig_csz.line(np.percentile(cszs, 50, axis=0), np.percentile(Fs, 50, axis=0)/std_Fs[dnm], line_color=alg[2], line_width=8, legend=alg[1])

rndlbl = bkm.Label(x=1.0, x_offset=-10, y=700, y_units='screen', text='Full Dataset MCMC', angle=90, angle_units='deg', text_font_size='30pt')
rndspan = bkm.Span(location = 1.0, dimension='height', line_width=8, line_color='black', line_dash='40 40')
fig_cput.add_layout(rndspan)
fig_cput.add_layout(rndlbl)

postprocess_plot(fig_cput, '22pt', location='bottom_left', glyph_width=40)
postprocess_plot(fig_csz, '22pt', location='bottom_left', glyph_width=40)

bkp.show(bkl.gridplot([[fig_cput, fig_csz]]))

