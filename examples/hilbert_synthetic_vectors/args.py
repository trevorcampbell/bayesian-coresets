import argparse

parser = argparse.ArgumentParser()

# example-specific arguments
parser.add_argument('alg_nm', type=str, choices=['FW', 'GIGA', 'OMP', 'IS', 'US'], help="The sparse non negative least squares algorithm to use: one of FW (Frank Wolfe), GIGA (Greedy Iterative Geodeic Ascent), OMP (Orthogonal Matching Pursuit), IS (Importance Sampling), US (Uniform Sampling)")
parser.add_argument('--data_num', type=int, default=1000, help="The number of synthetic data points")
parser.add_argument('--data_dim', type=int, default=100, help="The dimension of the synthetic data points, if applicable")
parser.add_argument('--data_type', type=str, default='normal', choices=['normal', 'axis'], help="Specifies the type of synthetic data to generate.")
parser.add_argument('--coreset_size_max', type=int, default=1000, help="The maximum coreset size to evaluate")
parser.add_argument('--coreset_num_sizes', type=int, default=100, help="The number of coreset sizes to evaluate")
parser.add_argument('--coreset_size_spacing', type=str, choices=['log', 'linear'], default='log', help="The spacing of coreset sizes to test")

# common arguments
parser.add_argument('--trial', type=int, help='The trial number (used to seed random number generation)')
parser.add_argument('--results_folder', type=str, default="results/", help="This script will save results in this folder. Default \"results/\"")
parser.add_argument('--verbosity', type=str, default="error", choices=['error', 'warning', 'critical', 'info', 'debug'], help="The verbosity level.")



# create a plot parser (with suppressed defaults)
# this will let us find out whether an argument has actually been specified
# see https://stackoverflow.com/questions/32056910/how-to-find-out-if-argparse-argument-has-been-actually-specified-on-command-line/45803037
arguments, unused = parser.parse_known_args()
argnames = list(vars(arguments))
plot_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
for arg in argnames: 
    plot_parser.add_argument('--'+arg)

# plotting arguments
plot_parser.add_argument('plot_x', type = str, help="The X axis of the plot")
plot_parser.add_argument('plot_y', type = str, help="The Y axis of the plot")
plot_parser.add_argument('--plot_x_type', type=str, choices=["linear","log"], default = "log", help = "Specifies the scale for the X-axis")
plot_parser.add_argument('--plot_y_type', type=str, choices=["linear","log"], default = "log", help = "Specifies the scale for the Y-axis.")
plot_parser.add_argument('--plot_legend', type=str, help = "Specifies the variable to create a legend for.")
plot_parser.add_argument('--plot_height', type=int, default=850, help = "Height of the plot's html canvas")
plot_parser.add_argument('--plot_width', type=int, default=850, help = "Width of the plot's html canvas")
plot_parser.add_argument('--plot_type', type=str, choices=['line', 'scatter'], default='scatter', help = "Type of plot to make")
plot_parser.add_argument('--plot_fontsize', type=str, default='32pt', help = "Font size for the figure, e.g., 32pt")
plot_parser.add_argument('--plot_toolbar', action='store_true', help = "Show the Bokeh toolbar")

