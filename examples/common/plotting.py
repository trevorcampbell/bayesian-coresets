from bokeh.models import FuncTickFormatter
import bokeh.palettes 
import numpy as np
import bokeh.plotting as bkp


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
'\u2079'];
var tick_power = Math.floor(Math.log10(tick));
var tick_mult = Math.pow(10, Math.log10(tick) - tick_power);
var ret = '';
if (tick_mult > 1.) {
  if (Math.abs(tick_mult - Math.round(tick_mult)) > 0.05){
    ret = tick_mult.toFixed(1) + '\u22C5';
  } else {
    ret = tick_mult.toFixed(0) +'\u22C5';
  }
}
ret += '10';
if (tick_power < 0){
  ret += '\u207B';
  tick_power = -tick_power;
}
var power_digits = [];
while (tick_power > 9){
  power_digits.push( tick_power - Math.floor(tick_power/10)*10 );
  tick_power = Math.floor(tick_power/10);
}
power_digits.push(tick_power)
for (var i = power_digits.length-1; i >= 0; i--){
  ret += trns[power_digits[i]];
}
return ret;
""")


pal = bokeh.palettes.colorblind['Colorblind'][8]
pl = [pal[0], pal[1], pal[3]]
pl.extend(pal[4:8])
pl.append('#d62728')
pal = pl

def preprocess_plot(fig, axis_font_size, log_scale_x, log_scale_y):
  fig.xaxis.axis_label_text_font_size= axis_font_size
  fig.xaxis.major_label_text_font_size= axis_font_size
  fig.yaxis.axis_label_text_font_size= axis_font_size
  fig.yaxis.major_label_text_font_size= axis_font_size
  if log_scale_y:
    fig.yaxis.formatter = logFmtr
  if log_scale_x:
    fig.xaxis.formatter = logFmtr

def postprocess_plot(fig, legend_font_size, orientation='vertical', location='top_right', glyph_width=80):
  fig.legend.label_text_font_size= legend_font_size
  fig.legend.orientation=orientation
  fig.legend.location=location
  fig.legend.glyph_width=glyph_width
  fig.legend.glyph_height=40
  fig.legend.spacing=5
  fig.xgrid.grid_line_color=None
  fig.ygrid.grid_line_color=None

def plot(arguments, df):
  fig = bkp.figure(title=arguments.plot_title, 
                 y_axis_type=arguments.plot_y_type, 
                 x_axis_type=arguments.plot_x_type, 
                 plot_width=arguments.plot_width,
                 plot_height=arguments.plot_height, 
                 x_axis_label=arguments.plot_x if arguments.plot_x_label is None else arguments.plot_x_label, 
                 y_axis_label=arguments.plot_y if arguments.plot_y_label is None else arguments.plot_y_label, 
                 toolbar_location='right' if arguments.plot_toolbar else None)

  preprocess_plot(fig, arguments.plot_fontsize, arguments.plot_x_type == 'log', arguments.plot_y_type == 'log')

  if arguments.plot_type == 'scatter':
    plotfunc = scatter
  else:
    plotfunc = line

  if arguments.plot_legend is not None:
    #iterate over groups
    i = 0
    for nm in sorted(df[arguments.plot_legend].unique()):
      tmpdf = df.loc[df[arguments.plot_legend] == nm]
      plotfunc(fig, tmpdf, arguments, clr = pal[i], legend = nm)
      i = i+1
  else:
    plotfunc(fig, df, arguments, clr = pal[0])

  postprocess_plot(fig, arguments.plot_fontsize, location='bottom_left', glyph_width=40)
  fig.legend.background_fill_alpha=0.
  fig.legend.border_line_alpha=0.
  fig.legend.visible = True
  bkp.show(fig)

def scatter(fig, df, arguments, clr = pal[0], legend = None):
  xy50 = df.groupby(arguments.groupby, as_index=False).quantile(.5)
  xy10 = df.groupby(arguments.groupby, as_index=False).quantile(.1)
  xy90 = df.groupby(arguments.groupby, as_index=False).quantile(.9)
  fig.scatter(xy50[arguments.plot_x], xy50[arguments.plot_y], color=clr, line_width=5, legend_label = legend)

  #err_xs = []
  #err_ys = []
  #for j in range(df.shape[0]):
  #  err_xs.append((xy10.iloc[j, arguments.plot_x], xy10.iloc[j, arguments.plot_x]))
  #  err_ys.append((xy10.iloc[j, arguments.plot_y], xy10.iloc[j, arguments.plot_x]))
  #  err_ys.append((np.percentile(y_all, 25, axis=0)[j], np.percentile(y_all, 75, axis=0)[j])) 

  #fig.multi_line(err_xs, err_ys, color=pal[i-1])

  #fig.scatter(xy50[arguments.plot_x], xy50[arguments.plot_y], color=pal[0], line_width=5)
  
def line(fig, df, arguments, clr = pal[0], legend = None):
  xy50 = df.groupby(arguments.groupby, as_index=False).quantile(.5)
  xy10 = df.groupby(arguments.groupby, as_index=False).quantile(.1)
  xy90 = df.groupby(arguments.groupby, as_index=False).quantile(.9)
  fig.line(xy50[arguments.plot_x], xy50[arguments.plot_y], color=clr, line_width=5, legend_label = legend)

  #err_xs = []
  #err_ys = []
  #for j in range(df.shape[0]):
  #  err_xs.append((xy10.iloc[j, arguments.plot_x], xy10.iloc[j, arguments.plot_x]))
  #  err_ys.append((xy10.iloc[j, arguments.plot_y], xy10.iloc[j, arguments.plot_x]))
  #  err_ys.append((np.percentile(y_all, 25, axis=0)[j], np.percentile(y_all, 75, axis=0)[j])) 

  #fig.multi_line(err_xs, err_ys, color=pal[i-1])

  #fig.scatter(xy50[arguments.plot_x], xy50[arguments.plot_y], color=pal[0], line_width=5)

def plot_gaussian(plot, mup, Sigp, Sig, color, dotsize, linewidth, dotalpha, linealpha, line_dash, name, num_pts_for_circle_approx = 100):
  plot.circle(mup[0], mup[1], color=color, size=dotsize, alpha=dotalpha)
  t = np.linspace(0., 2*np.pi, num_pts_for_circle_approx)
  t = np.array([np.cos(t), np.sin(t)])
  t = 3*np.linalg.cholesky(Sigp+Sig).dot(t) + mup[:, np.newaxis]
  plot.line(t[0, :], t[1, :], color=color, line_width=linewidth, alpha=linealpha, line_dash=line_dash, legend=name)

def plot_meanstd(plot, x, ys, color, linewidth, alpha, line_dash, name):
  plot.line(x, ys.mean(axis=0), color=color, line_width=linewidth, line_dash=line_dash, legend=nm)
  plot.patch(np.hstack((x, x[::-1])), np.hstack(( ys.mean(axis=0)-ys.std(axis=0), (ys.mean(axis=0)+ys.std(axis=0))[::-1] )), color=color, line_width=linewidth/2, line_dash=line_dash, alpha=alpha, legend=nm)

def plot_medianquartiles(plot, x, ys, color, linewidth, alpha, line_dash, name):
  ys25 = np.percentile(ys, 49, axis=0)
  ys50 = np.percentile(ys, 50, axis=0)
  ys75 = np.percentile(ys, 51, axis=0)
  plot.line(x, ys25, color=color, line_width=linewidth, line_dash=line_dash, legend=nm)
  plot.line(x, ys50, color=color, line_width=linewidth, line_dash=line_dash, legend=nm)
  plot.line(x, ys75, color=color, line_width=linewidth, line_dash=line_dash, legend=nm)
  #plot.patch(np.hstack((x, x[::-1])), np.hstack(( ys25, ys75[::-1] )), color=color, line_width=linewidth/2, line_dash=line_dash, alpha=alpha, legend=nm)

def plot_gaussian_projected2d(dim, mu, sig, plot,
                              color=pal[7], dotsize=20, linewidth=10, dotalpha=1, linealpha=1, line_dash='solid', name="POST", 
                              num_pts_for_circle_approx = 10,seed = 1):
  #set seed:
  np.random.seed(int(seed))

  #get two vectors that form an orthonormal basis: 
  x = np.random.randn(dim,1)
  x = x/np.linalg.norm(x)
  
  y = np.random.randn(dim,1)
  y = y-x@x.T@y
  y = y/np.linalg.norm(y)
  
  #make the orthonormal basis (as a matrix)
  A = np.append(x,y, axis=1)
  
  #project the mean mu and variance sig onto this basis
  mu = A.T@mu
  sig = A.T@sig@A 

  #plot the result
  plot_gaussian(plot,mu,sig,0,color=color,dotsize=dotsize,linewidth=linewidth,dotalpha=dotalpha,linealpha=linealpha,
  line_dash=line_dash,name=name, num_pts_for_circle_approx=num_pts_for_circle_approx)


