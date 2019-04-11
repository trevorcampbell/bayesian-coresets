import bokeh.plotting as bkp
import bokeh.io as bki
import bokeh.layouts as bkl
from bokeh.models import FuncTickFormatter
import bokeh.palettes 
import numpy as np

def plot_gaussian(plot, mup, Sigp, Sig, color, dotsize, linewidth, dotalpha, linealpha, name):
  plot.circle(mup[0], mup[1], color=color, size=dotsize, alpha=dotalpha, legend=name)
  t = np.linspace(0., 2*np.pi, 100)
  t = np.array([np.cos(t), np.sin(t)])
  t = 3*np.linalg.cholesky(Sigp+Sig).dot(t) + mup[:, np.newaxis]
  plot.line(t[0, :], t[1, :], color=color, line_width=linewidth, alpha=linealpha, legend=name)

#logFmtr = FuncTickFormatter(code="return Math.log10(tick)")
logFmtr = FuncTickFormatter(code="""
var trns = [
'\u2070',
'\u00B9',
'\u00B2',
'\u00B3',
'\u2074',
'\u2075',
'\u2076',
'\u2077',
'\u2078',
'\u2079']
if (Math.log10(tick) < 0){
  return '10\u207B'+trns[Math.round(Math.abs(Math.log10(tick)))];
} else {
  return '10'+trns[Math.round(Math.abs(Math.log10(tick)))];
}
""")

pal = bokeh.palettes.colorblind['Colorblind'][8]

null_font_size='0pt'
axis_font_size='40pt'

figs = []

#command that generated the results files
#np.savez('results.npz', x=x, th0=th0, Sig0=Sig0, Sig=Sig, mup=mup, Sigp=Sigp, lambdas=lambdas, 
#                        w_l1=w_l1, w_l1_post=w_l1_post, w_g=w_g, w_g_post=w_g_post, 
#                        muw_l1=muw_l1, muw_l1_post=muw_l1_post, muw_g=muw_g, muw_g_post = muw_g_post,
#                        Sigw_l1=Sigw_l1, Sigw_l1_post = Sigw_l1_post, Sigw_g=Sigw_g, Sigw_g_post=Sigw_g_post,
#                        kl_g=kl_g, kl_g_post=kl_g_post, kl_l1=kl_l1, kl_l1_post=kl_l1_post)

res = np.load('results_EGUS.npz')
x = res['x']
w = res['w']
Sig = res['Sig']
mup = res['mup']
Sigp = res['Sigp']
muw = res['muw']
Sigw = res['Sigw']
rklw = res['rklw']
fklw = res['rklw']

for m in range(w.shape[0]):
  fig = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
  
  fig.scatter(x[:, 0], x[:, 1], fill_color='black', alpha=0.1)
  fig.scatter(x[:, 0], x[:, 1], fill_color='black', size=20*w[m,:]/w[m,:].max())
  plot_gaussian(fig, mup, Sigp, Sig, 'blue', 5, 1, 1, 1, 'True')
  plot_gaussian(fig, muw[m,:], Sigw[m,:], Sig, 'green', 5, 1, 1, 1, 'Coreset')
  figs.append([fig])

bkp.show(bkl.gridplot(figs))

