import numpy as np
import hilbertcoresets as hc
import bokeh.plotting as bkp
import bokeh.io as bki
import bokeh.layouts as bkl
from bokeh.models import FuncTickFormatter
import bokeh.palettes 


N = 10000
D = 50

X = np.random.randn(N, D)
XS = X.sum(axis=0)


Ms = np.linspace(1, 1000, 1000, dtype=np.int32)
giga_fast = hc.GIGA(X, 'fast')
fw_fast = hc.FrankWolfe(X, 'fast')
giga_acc = hc.GIGA(X, 'accurate')
fw_acc = hc.FrankWolfe(X, 'accurate')
ims = hc.ImportanceSampling(X)
rnd = hc.RandomSubsampling(X)

algs = [giga_fast, fw_fast, giga_acc, fw_acc, ims, rnd]
anms = ['GIGA(fast)', 'FW(fast)', 'GIGA(accurate)', 'FW(accurate)', 'IS', 'RND']
pal = bokeh.palettes.colorblind['Colorblind'][8]

fig = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Err', x_axis_label='M', plot_width=1250, plot_height=1250)
axis_font_size='30pt'
fig.xaxis.axis_label_text_font_size= axis_font_size
fig.xaxis.major_label_text_font_size= axis_font_size
fig.yaxis.axis_label_text_font_size= axis_font_size
fig.yaxis.major_label_text_font_size= axis_font_size


for anm, alg, clr in zip(anms, algs, pal[:6]):
  err = np.zeros(len(Ms))
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


