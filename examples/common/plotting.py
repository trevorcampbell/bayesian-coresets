from bokeh.models import FuncTickFormatter
import bokeh.palettes 
import numpy as np


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
power_digits = []
while (tick_power > 9){
  power_digits.push( tick_power - Math.floor(tick_power/10)*10 )
  tick_power = Math.floor(tick_power/10)
}
power_digits.push(tick_power)
for (i = power_digits.length-1; i >= 0; i--){
  ret += trns[power_digits[i]];
}
return ret;
""")


pal = bokeh.palettes.colorblind['Colorblind'][8]
pl = [pal[0], pal[1], pal[3]]
pl.extend(pal[4:8])
pl.append('#d62728')
pal = pl



def plot_gaussian(plot, mup, Sigp, Sig, color, dotsize, linewidth, dotalpha, linealpha, line_dash, name):
  plot.circle(mup[0], mup[1], color=color, size=dotsize, alpha=dotalpha)
  t = np.linspace(0., 2*np.pi, 100)
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


def preprocess_plot(fig, axis_font_size, log_scale_x, log_scale_y):
  fig.xaxis.axis_label_text_font_size= axis_font_size
  fig.xaxis.major_label_text_font_size= axis_font_size
  fig.yaxis.axis_label_text_font_size= axis_font_size
  fig.yaxis.major_label_text_font_size= axis_font_size
  if log_scale_y:
    fig.yaxis.formatter = logFmtr
  if log_scale_x:
    fig.xaxis.formatter = logFmtr
  #fig.toolbar.logo = None
  #fig.toolbar_location = None

def postprocess_plot(fig, legend_font_size, orientation='vertical', location='top_right', glyph_width=80):
  fig.legend.label_text_font_size= legend_font_size
  fig.legend.orientation=orientation
  fig.legend.location=location
  fig.legend.glyph_width=glyph_width
  fig.legend.glyph_height=40
  fig.legend.spacing=5
  fig.xgrid.grid_line_color=None
  fig.ygrid.grid_line_color=None

