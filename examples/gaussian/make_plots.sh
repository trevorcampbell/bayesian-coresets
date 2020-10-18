#!/bin/bash

python3 main.py plot Ms rklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Reverse KL"
python3 main.py plot csizes rklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Reverse KL"
python3 main.py plot cputs rklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Reverse KL"

python3 main.py plot Ms fklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Forward KL"
python3 main.py plot csizes fklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Forward KL"
python3 main.py plot cputs fklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Forward KL"





