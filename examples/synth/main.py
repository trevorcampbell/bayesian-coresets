import numpy as np
import hilbertcoresets as hc
import bokeh.plotting as bkp
import bokeh.io as bki
import bokeh.layouts as bkl
from bokeh.models import FuncTickFormatter
import bokeh.palettes 
import time


##########################################
## Test 1: 1M 100-dimensional gaussian data
##########################################
N = 10000
D = 10
Ms = np.linspace(1, 10000, 10000, dtype=np.int32)
n_trials = 50
axis_font_size='30pt'

fig_err = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Err', x_axis_label='M', plot_width=1250, plot_height=1250)
fig_err.xaxis.axis_label_text_font_size= axis_font_size
fig_err.xaxis.major_label_text_font_size= axis_font_size
fig_err.yaxis.axis_label_text_font_size= axis_font_size
fig_err.yaxis.major_label_text_font_size= axis_font_size

fig_cost = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Time (ms) & #Ops', x_axis_label='M', plot_width=1250, plot_height=1250)
fig_cost.xaxis.axis_label_text_font_size= axis_font_size
fig_cost.xaxis.major_label_text_font_size= axis_font_size
fig_cost.yaxis.axis_label_text_font_size= axis_font_size
fig_cost.yaxis.major_label_text_font_size= axis_font_size



anms = ['GIGA(A)', 'GIGA(L)', 'FW', 'RND']
pal = bokeh.palettes.colorblind['Colorblind'][4]
for anm, clr in zip(anms, pal):
  err = np.zeros((n_trials, Ms.shape[0]))
  nfunc = np.zeros((n_trials, Ms.shape[0]))
  cput = np.zeros((n_trials, Ms.shape[0]))
  for tr in range(n_trials):
    X = np.random.randn(N, D)
    XS = X.sum(axis=0)
    alg = None
    if anm == 'GIGA(A)' or anm == 'GIGA(L)':
      alg = hc.GIGA(X)
    elif anm == 'FW'#initialize algorithms
      alg = hc.FrankWolfe(X)
    else:
      alg = hc.RandomSubsampling(X) 

    for m, M in enumerate(Ms):
      t0 = time.clock()
      alg.run(M)
      tf = time.clock()
      cput[tr, m] = tf-t0 + cput[tr, m-1] if m > 0 else tf-t0
      wts = alg.weights()
      err[tr, m] = np.sqrt((((wts[:, np.newaxis]*X).sum(axis=0) - XS)**2).sum())
      nfunc[tr, m] = alg.get_num_ops()
 
  fig_err.line(Ms, np.percentile(err, 50, axis=0), line_color=clr, line_width=4, legend=anm)
  fig_cost.line(Ms, np.percentile(nfunc, 50, axis=0), line_color=clr, line_width=4, legend=anm)
  fig_cost.line(Ms, np.percentile(cput, 50, axis=0), line_color=clr, line_width=4, line_dash='dashed')

fig_err.legend.label_text_font_size= '16pt'
fig_err.legend.glyph_width=40
fig_err.legend.glyph_height=40
fig_err.legend.spacing=20

fig_cost.legend.label_text_font_size= '16pt'
fig_cost.legend.glyph_width=40
fig_cost.legend.glyph_height=40
fig_cost.legend.spacing=20

 
bkp.show(bkl.gridplot([[fig_err, fig_cost]]))


