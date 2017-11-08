import numpy as np
import hilbertcoresets as hc
import bokeh.plotting as bkp
import bokeh.io as bki
import bokeh.layouts as bkl
from bokeh.models import FuncTickFormatter
import bokeh.palettes 


#10^5 data 100-dimensional gaussian data
N = 100000
D = 100
X = np.random.randn(N, D)
XS = X.sum(axis=0)

#sweep over iterations 1->1000
Ms = np.linspace(1, 1000, 1000, dtype=np.int32)
#initialize algorithms
giga = hc.GIGA(X)
fw = hc.FrankWolfe(X)
rnd = hc.RandomSubsampling(X)
algs = [giga, fw, rnd]
anms = ['GIGA', 'FW', 'RND']

#initialize figures / color palette
pal = bokeh.palettes.colorblind['Colorblind'][3]
axis_font_size='30pt'

fig_err = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Err', x_axis_label='M', plot_width=1250, plot_height=1250)
fig_err.xaxis.axis_label_text_font_size= axis_font_size
fig_err.xaxis.major_label_text_font_size= axis_font_size
fig_err.yaxis.axis_label_text_font_size= axis_font_size
fig_err.yaxis.major_label_text_font_size= axis_font_size

fig_nfunc = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='nFunc', x_axis_label='M', plot_width=1250, plot_height=1250)
fig_nfunc.xaxis.axis_label_text_font_size= axis_font_size
fig_nfunc.xaxis.major_label_text_font_size= axis_font_size
fig_nfunc.yaxis.axis_label_text_font_size= axis_font_size
fig_nfunc.yaxis.major_label_text_font_size= axis_font_size

#for every algorithm to be tested
for anm, alg, clr in zip(anms, algs, pal):
  err = np.zeros(len(Ms))
  nfunc = np.zeros(len(Ms))
  if anm != 'GIGA':
    #for FW and RND, just run one sweep over M and plot the result at the end
    for m, M in enumerate(Ms):
      alg.run(M)
      wts = alg.weights()
      err[m] = np.sqrt((((wts[:, np.newaxis]*X).sum(axis=0) - XS)**2).sum())
      nfunc[m] = alg.

    #fig.line(Ms, err, line_color=clr, line_dash='dashed', line_dash_offset=int(np.random.rand()*100), line_width=4, legend=anm)
    fig_err.line(Ms, err, line_color=clr, line_width=4, legend=anm)
    fig_nfunc.line(Ms, nfunc, line_color=clr, line_width=4, legend=anm)
  else:
    #for GIGA, try all the different search methods. For adaptive, need to restart at each M (since choice of whether to build tree is based on M)
    for search_method in ['linear', 'tree', 'adaptive']:
      for m, M in enumerate(Ms):
        alg.run(M)
        wts = alg.weights()
        err[m] = np.sqrt((((wts[:, np.newaxis]*X).sum(axis=0) - XS)**2).sum())

      fig.line(Ms, err, line_color=clr, line_dash='dashed', line_dash_offset=int(np.random.rand()*100), line_width=4, legend=anm)

fig.legend.label_text_font_size= '16pt'
fig.legend.glyph_width=40
fig.legend.glyph_height=40
fig.legend.spacing=20

 
bkp.show(fig)


