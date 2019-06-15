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

fig_cput = bkp.figure(y_axis_type='log', y_axis_label='Normalized Fisher Information Distance', x_axis_type='log', x_axis_label='Relative Total CPU Time', x_range=(.05, 1.1), plot_width=1250, plot_height=1250)
fig_csz = bkp.figure(y_axis_type='log', y_axis_label='Normalized Fisher Information Distance', x_axis_type='log', x_axis_label='Coreset Size', plot_width=1250, plot_height=1250)

preprocess_plot(fig_cput, '32pt', False, True)
preprocess_plot(fig_csz, '32pt', True, True)


for didx, dnm in enumerate(dnames):
  res = np.load('results/'+dnm  + '_results.npz')
  Fs = res['Fs']
  cputs = res['cputs']
  cputs_full = res['cputs_full']
  csizes = res['csizes']
  anms = res['anms']

  for aidx, anm in enumerate(anms):
    if anm == 'FW':
      clr = pal[1]
    elif anm == 'GIGA':
      clr = pal[0]
    else:
      clr = pal[2]

    fig_cput.line(np.percentile(cputs[aidx,:,:], 50, axis=0)/np.percentile(cputs_full, 50, axis=0), np.percentile(Fs[aidx, :, :], 50, axis=0)/np.percentile(Fs[2, :, :], 50), line_color=clr, line_width=8, legend=anm)
    fig_csz.line(np.percentile(csizes[aidx,:,:], 50, axis=0), np.percentile(Fs[aidx, :, :], 50, axis=0)/np.percentile(Fs[2, :, :], 50), line_color=clr, line_width=8, legend=anm)
    
rndlbl = bkm.Label(x=1.0, x_offset=-10, y=700, y_units='screen', text='Full Dataset MCMC', angle=90, angle_units='deg', text_font_size='30pt')
rndspan = bkm.Span(location = 1.0, dimension='height', line_width=8, line_color='black', line_dash='40 40')
fig_cput.add_layout(rndspan)
fig_cput.add_layout(rndlbl)


postprocess_plot(fig_cput, '22pt', location='bottom_left', glyph_width=40)
postprocess_plot(fig_csz, '22pt', location='bottom_left', glyph_width=40)


bkp.show(bkl.gridplot([[fig_cput, fig_csz]]))

