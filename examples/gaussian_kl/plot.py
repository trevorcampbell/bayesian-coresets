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

res = np.load('results.npz')
x = res['x']
w_l1 = res['w_l1_post']
w_g = res['w_g_post']
Sig = res['Sig']
mup = res['mup']
Sigp = res['Sigp']
muw_l1 = res['muw_l1_post']
Sigw_l1 = res['Sigw_l1_post']
muw_g = res['muw_g_post']
Sigw_g = res['Sigw_g_post']

for m in range(w_l1.shape[0]):
  fig_l1 = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
  fig_g = bkp.figure(x_range=(-5,5), y_range=(-5,5), plot_width=750, plot_height=750)
  
  fig_l1.scatter(x[:, 0], x[:, 1], fill_color='black', alpha=0.1)
  fig_l1.scatter(x[:, 0], x[:, 1], fill_color='black', size=20*w_l1[m,:]/w_l1[m,:].max())
  plot_gaussian(fig_l1, mup, Sigp, Sig, 'blue', 5, 1, 1, 1, 'True')
  plot_gaussian(fig_l1, muw_l1[m,:], Sigw_l1[m,:], Sig, 'green', 5, 1, 1, 1, 'Coreset')
  fig_g.scatter(x[:, 0], x[:, 1], fill_color='black', alpha=0.1)
  fig_g.scatter(x[:, 0], x[:, 1], fill_color='black', size=20*w_g[m,:]/w_g[m,:].max())
  plot_gaussian(fig_g, mup, Sigp, Sig, 'blue', 5, 1, 1, 1, 'True')
  plot_gaussian(fig_g, muw_g[m,:], Sigw_g[m,:], Sig, 'green', 5, 1, 1, 1, 'Coreset')
  
  figs.append([fig_l1, fig_g])

bkp.show(bkl.gridplot(figs))

####for each alg, create a horizontal row of figures (one for each value of M)
####for each horizontal row, run T tests where we successively build up a coreset
####plot one of the tests in green with coreset pts, true posterior in blue, weights in shaded gray
###for r in results:
###  nm, res = r
###  figs.append([])
###  n_M = len(res[0])
###  for m in range(n_M):
###    figs[-1].append() #, title=(nm if m == 0 else None))) 
###    figs[-1][-1].xaxis.axis_label_text_font_size= axis_font_size
###    figs[-1][-1].xaxis.major_label_text_font_size= axis_font_size
###    figs[-1][-1].yaxis.axis_label_text_font_size= axis_font_size
###    figs[-1][-1].yaxis.major_label_text_font_size= axis_font_size
###    figs[-1][-1].toolbar.logo = None
###    figs[-1][-1].toolbar_location = None
###    figs[-1][-1].circle(x[:,0], x[:, 1], color='black', size=10, alpha=0.1)#, legend=('Data' if m == n_M-1 else None))
###  for t in range(len(res)):
###    for m in range(len(res[t])):
###      if len(res[t][m]) == 4:
###        widx = res[t][m][2]
###        weights = res[t][m][3]
###        figs[-1][m].circle(x[widx, 0], x[widx, 1], color='black', size=40*np.maximum(0.2, weights[widx]/weights[widx].max()))
###        if m == n_M - 1:
###          figs[-1][m].circle(x[weights.argmax(), 0], x[weights.argmax(), 1], color='black', size=20, legend='Coreset')
###  for t in range(len(res)):
###    for m in range(len(res[t])): 
###      plot_gaussian(figs[-1][m], res[t][m][0], res[t][m][1], Sig, 'forestgreen', 20, 2, 0.5 , 0.5 , ('Approx' if m == n_M-1 else None))
###      plot_gaussian(figs[-1][m], emup, eSigp, Sig, 'royalblue', 20, 4, 0.75, 0.75, ('Exact' if m == n_M-1 else None))
###      
###
###      figs[-1][m].legend.label_text_font_size= '32pt'
###      figs[-1][m].legend.glyph_width=40
###      figs[-1][m].legend.glyph_height=40
###      figs[-1][m].legend.spacing=20
###      figs[-1][m].legend.orientation='horizontal'
###      figs[-1][m].xgrid.grid_line_color=None
###      figs[-1][m].ygrid.grid_line_color=None
###
#f = open('test2.npy', 'rb')
#results = cpk.load(f)
#f.close()
#
#
##plot the KL divergence vs M for all algs
##in this plot, make sure to resample x and mu0 for each trial
#fig = bkp.figure(title=None, x_axis_label='M', y_axis_label='KL Divergence', y_axis_type='log', plot_width=1000, plot_height=800, x_range=(-40, 1080))
#fig.xaxis.axis_label_text_font_size= axis_font_size
#fig.xaxis.major_label_text_font_size= axis_font_size
#fig.yaxis.axis_label_text_font_size= axis_font_size
#fig.yaxis.major_label_text_font_size= axis_font_size
#fig.yaxis.formatter = logFmtr
#fig.toolbar.logo = None
#fig.toolbar_location = None
#print('KL vs M')
#for nm, clr, kl in results:
#  print(nm)
#  #kl_95 = np.percentile(kl, 95, axis=0)
#  #kl_75 = np.percentile(kl, 75, axis=0)
#  kl_50 = np.percentile(kl, 50, axis=0)
#  #kl_25 = np.percentile(kl, 25, axis=0)
#  #kl_5 = np.percentile(kl, 5, axis=0)
#  fig.line(np.arange(1, kl.shape[1]+1)[::20], kl_50[::20], color=post_clrs[nm], line_width=4) #, legend=post_nms[nm])
#  if post_mrkrs[nm] == 'circle':
#    fig.circle(np.arange(1, kl.shape[1]+1)[::20], kl_50[::20], fill_color=post_clrs[nm],line_color=None, size=10)
#  elif post_mrkrs[nm] == 'square':
#    fig.square(np.arange(1, kl.shape[1]+1)[::20], kl_50[::20], fill_color=post_clrs[nm],line_color=None, size=10)
#  elif post_mrkrs[nm] == 'diamond':
#    fig.diamond(np.arange(1, kl.shape[1]+1)[::20], kl_50[::20], fill_color=post_clrs[nm],line_color=None, size=20)
#  else:
#    fig.triangle(np.arange(1, kl.shape[1]+1)[::20], kl_50[::20], fill_color=post_clrs[nm],line_color=None, size=10)
#   
# 
#  #fig.line(np.arange(1, kl.shape[1]+1), kl_75, color=post_clrs[nm], line_width=2, line_dash='dashed')
#  #fig.line(np.arange(1, kl.shape[1]+1), kl_25, color=post_clrs[nm], line_width=2, line_dash='dashed')
#  #fig.line(np.arange(1, kl.shape[1]+1), kl_95, color=post_clrs[nm], line_width=2, line_dash='dotted')
#  #fig.line(np.arange(1, kl.shape[1]+1), kl_5, color=post_clrs[nm], line_width=2, line_dash='dotted')
##fig.legend.label_text_font_size= '40pt'
##fig.legend.orientation='horizontal'
##fig.legend.glyph_width=40
##fig.legend.glyph_height=40
##fig.legend.spacing=20
#figs.append([fig])
#
#f = open('test3.npy', 'rb')
#results = cpk.load(f)
#f.close()
#
##plot the KL divergence vs D for all algs
##in this plot, make sure to resample x and mu0 for each trial
#fig = bkp.figure(title=None, x_axis_label='Projection Dimension', y_axis_label='KL Divergence', y_axis_type='log', plot_width=1000, plot_height=800, x_range=(-2, 42))
#fig.xaxis.axis_label_text_font_size= axis_font_size
#fig.xaxis.major_label_text_font_size= axis_font_size
#fig.yaxis.axis_label_text_font_size= axis_font_size
#fig.yaxis.major_label_text_font_size= axis_font_size
#fig.yaxis.formatter = logFmtr
#fig.toolbar.logo = None
#fig.toolbar_location = None
#print('KL vs D')
#for nm, clr, kl in results:
#  print(nm)
#  #kl_95 = np.percentile(kl, 95, axis=0)
#  #kl_75 = np.percentile(kl, 75, axis=0)
#  kl_50 = np.percentile(kl, 50, axis=0)
#  #kl_25 = np.percentile(kl, 25, axis=0)
#  #kl_5 = np.percentile(kl, 5, axis=0)
#  if nm == 'Importance Sampling (F)' or nm == 'Frank-Wolfe (F)':
#    fig.line(np.arange(1, kl.shape[1]+1), np.ones(kl.shape[1])*np.percentile(kl_50, 50, axis=0), color=post_clrs[nm], line_width=4) #, legend=post_nms[nm])
#    if post_mrkrs[nm] == 'circle':
#      fig.circle(np.arange(1, kl.shape[1]+1), np.ones(kl.shape[1])*np.percentile(kl_50, 50, axis=0), fill_color=post_clrs[nm],line_color=None, size=10)
#    elif post_mrkrs[nm] == 'square':
#      fig.square(np.arange(1, kl.shape[1]+1), np.ones(kl.shape[1])*np.percentile(kl_50, 50, axis=0), fill_color=post_clrs[nm],line_color=None, size=10)
#    elif post_mrkrs[nm] == 'diamond':
#      fig.diamond(np.arange(1, kl.shape[1]+1), np.ones(kl.shape[1])*np.percentile(kl_50, 50, axis=0), fill_color=post_clrs[nm],line_color=None, size=20)
#    else:
#      fig.triangle(np.arange(1, kl.shape[1]+1), np.ones(kl.shape[1])*np.percentile(kl_50, 50, axis=0), fill_color=post_clrs[nm],line_color=None, size=10)
#  elif nm == 'Projected Importance Sampling (F)' or nm == 'Projected Frank-Wolfe (F)':
#    fig.line(np.arange(1, kl.shape[1]+1), kl_50, color=post_clrs[nm], line_width=4) #, legend=post_nms[nm])
#    if post_mrkrs[nm] == 'circle':
#      fig.circle(np.arange(1, kl.shape[1]+1), kl_50, fill_color=post_clrs[nm],line_color=None, size=10)
#    elif post_mrkrs[nm] == 'square':
#      fig.square(np.arange(1, kl.shape[1]+1), kl_50, fill_color=post_clrs[nm],line_color=None, size=10)
#    elif post_mrkrs[nm] == 'diamond':
#      fig.diamond(np.arange(1, kl.shape[1]+1), kl_50, fill_color=post_clrs[nm],line_color=None, size=20)
#    else:
#      fig.triangle(np.arange(1, kl.shape[1]+1), kl_50, fill_color=post_clrs[nm],line_color=None, size=10)
#  
##fig.legend.label_text_font_size= '40pt'
##fig.legend.orientation='horizontal'
##fig.legend.glyph_width=40
##fig.legend.glyph_height=40
##fig.legend.spacing=20
#figs.append([fig])
#
#
##f = open('test3.npy', 'rb')
##fwnm, fwclr, fwkl, nm, clr, kl, nm2, clr2, kl2 = cpk.load(f)
##f.close()
##
###plot the KL divergence vs sketching dimension at fixed M
##fig = bkp.figure(title=None, x_axis_label='D', y_axis_label='KL Divergence', y_axis_type='log', plot_width=1000, plot_height=800, x_range=(-2, 42), y_range=(.005, 5000))
##fig.xaxis.axis_label_text_font_size= axis_font_size
##fig.xaxis.major_label_text_font_size= axis_font_size
##fig.yaxis.axis_label_text_font_size= axis_font_size
##fig.yaxis.major_label_text_font_size= axis_font_size
##fig.yaxis.formatter = logFmtr
##fig.toolbar.logo = None
##fig.toolbar_location = None
##
##kl_95 = np.percentile(kl, 95, axis=0)
##kl_75 = np.percentile(kl, 75, axis=0)
##kl_50 = np.percentile(kl, 50, axis=0)
##kl_25 = np.percentile(kl, 25, axis=0)
##kl_5 = np.percentile(kl, 5, axis=0)
##fig.line(np.arange(1, kl.shape[1]+1), kl_50, color=post_clrs[nm], line_width=4) #, legend=post_nms[nm])
##fig.line(np.arange(1, kl.shape[1]+1), kl_75, color=post_clrs[nm], line_width=2, line_dash='dashed')
##fig.line(np.arange(1, kl.shape[1]+1), kl_25, color=post_clrs[nm], line_width=2, line_dash='dashed')
##fig.line(np.arange(1, kl.shape[1]+1), kl_95, color=post_clrs[nm], line_width=2, line_dash='dotted')
##fig.line(np.arange(1, kl.shape[1]+1), kl_5 , color=post_clrs[nm], line_width=2, line_dash='dotted')
##
##kl2_95 = np.percentile(kl2, 95, axis=0)
##kl2_75 = np.percentile(kl2, 75, axis=0)
##kl2_50 = np.percentile(kl2, 50, axis=0)
##kl2_25 = np.percentile(kl2, 25, axis=0)
##kl2_5 =  np.percentile(kl2, 5, axis=0)
##fig.line(np.arange(1, kl2.shape[1]+1), kl2_50, color=post_clrs[nm2], line_width=4) #, legend=post_nms[nm2])
##fig.line(np.arange(1, kl2.shape[1]+1), kl2_75, color=post_clrs[nm2], line_width=2, line_dash='dashed')
##fig.line(np.arange(1, kl2.shape[1]+1), kl2_25, color=post_clrs[nm2], line_width=2, line_dash='dashed')
##fig.line(np.arange(1, kl2.shape[1]+1), kl2_95, color=post_clrs[nm2], line_width=2, line_dash='dotted')
##fig.line(np.arange(1, kl2.shape[1]+1), kl2_5 , color=post_clrs[nm2], line_width=2, line_dash='dotted')
##
##fig.line(np.arange(1, kl.shape[1]+1), np.percentile(fwkl, 50)*np.ones(kl.shape[1]), color=post_clrs[fwnm], line_width=4) #, legend=post_nms[fwnm])
##fig.line(np.arange(1, kl.shape[1]+1), np.percentile(fwkl, 75)*np.ones(kl.shape[1]), color=post_clrs[fwnm], line_width=2, line_dash='dashed')
##fig.line(np.arange(1, kl.shape[1]+1), np.percentile(fwkl, 25)*np.ones(kl.shape[1]), color=post_clrs[fwnm], line_width=2, line_dash='dashed')
##fig.line(np.arange(1, kl.shape[1]+1), np.percentile(fwkl, 95)*np.ones(kl.shape[1]), color=post_clrs[fwnm], line_width=2, line_dash='dotted')
##fig.line(np.arange(1, kl.shape[1]+1), np.percentile(fwkl, 5) *np.ones(kl.shape[1]), color=post_clrs[fwnm], line_width=2, line_dash='dotted')
##
###fig.legend.label_text_font_size= '40pt'
###fig.legend.orientation='horizontal'
###fig.legend.glyph_width=40
###fig.legend.glyph_height=40
###fig.legend.spacing=20
##
##figs.append([fig])
#
###plot the legend
#fig = bkp.figure(title=None, x_axis_label='D', y_axis_label='KL Divergence', y_axis_type='log', plot_width=3000, plot_height=800)
#
#for nm in ['Random', 'Uniform', 'Importance Sampling (F)', 'Frank-Wolfe (F)', 'Projected Importance Sampling (F)', 'Projected Frank-Wolfe (F)']:
#  fig.line(np.arange(1, 10), np.zeros(9), line_width=8, color=post_clrs[nm], legend=post_nms[nm])
#  if post_mrkrs[nm] == 'circle':
#    fig.circle(np.arange(1, 10), np.zeros(9), fill_color=post_clrs[nm],line_color=None, size=10, legend=post_nms[nm])
#  elif post_mrkrs[nm] == 'square':
#    fig.square(np.arange(1, 10), np.zeros(9), fill_color=post_clrs[nm],line_color=None, size=10, legend=post_nms[nm])
#  elif post_mrkrs[nm] == 'diamond':
#    fig.diamond(np.arange(1, 10), np.zeros(9), fill_color=post_clrs[nm],line_color=None, size=20, legend=post_nms[nm])
#  else:
#    fig.triangle(np.arange(1, 10), np.zeros(9), fill_color=post_clrs[nm],line_color=None, size=10, legend=post_nms[nm])
#fig.legend.label_text_font_size= '60pt'
#fig.legend.orientation='horizontal'
#fig.legend.location='top_left'
#fig.legend.glyph_width=80
#fig.legend.glyph_height=80
#fig.legend.spacing=40
#fig.xgrid.grid_line_color=None
#fig.ygrid.grid_line_color=None
#figs.append([fig])
#
##fctr = 0
##for f in figs:
##  for ff in f:
##    bki.export_png(ff, 'fig'+'%03d'%fctr+'.png')
##    fctr += 1
