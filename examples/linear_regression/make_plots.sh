#!/bin/bash


python3 main.py plot Ms rklw --summarize trial proj_dim --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Reverse KL"
python3 main.py plot csizes rklw --summarize trial proj_dim --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Reverse KL"
python3 main.py plot cputs rklw --summarize trial proj_dim --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Reverse KL"

python3 main.py plot Ms fklw --summarize trial proj_dim --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Forward KL"
python3 main.py plot csizes fklw --summarize trial proj_dim --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Forward KL"
python3 main.py plot cputs fklw --summarize trial proj_dim --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Forward KL"

python3 main.py plot Ms mu_errs --summarize trial proj_dim --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Relative Mean Error"
python3 main.py plot csizes mu_errs --summarize trial proj_dim --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Relative Mean Error"
python3 main.py plot cputs mu_errs --summarize trial proj_dim --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Relative Mean Error"

python3 main.py plot Ms Sig_errs --summarize trial proj_dim --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Relative Covariance Error"
python3 main.py plot csizes Sig_errs --summarize trial proj_dim --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Relative Covariance Error"
python3 main.py plot cputs Sig_errs --summarize trial proj_dim --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Relative Covariance Error"





