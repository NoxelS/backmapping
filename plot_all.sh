#!/bin/bash

mkdir -p hist_plots

sbatch --wrap="python src/plot_hist.py -c relu_run -n hist_plots/relu_run.pdf"
sbatch --wrap="python src/plot_hist.py -c do_run -n hist_plots/do_run.pdf"
sbatch --wrap="python src/plot_hist.py -c fe_run -n hist_plots/fe_run.pdf"
sbatch --wrap="python src/plot_hist.py -c activation_run -n hist_plots/activation_run.pdf"
sbatch --wrap="python src/plot_hist.py -c aug_run -n hist_plots/aug_run.pdf"
sbatch --wrap="python src/plot_hist.py -c bs_test -n hist_plots/bs_run.pdf"
sbatch --wrap="python src/plot_hist.py -c lr_run -n hist_plots/lr_run.pdf"
sbatch --wrap="python src/plot_hist.py -c capacity -n hist_plots/capacity_run.pdf"
sbatch --wrap="python src/plot_hist.py -c loss_test -n hist_plots/loss_run.pdf"
sbatch --wrap="python src/plot_hist.py -c lr_run_0.00000 -n hist_plots/lr_run_small.pdf"
sbatch --wrap="python src/plot_hist.py -fs training_history_lr_run_0.000007_94.csv -n hist_plots/lr_run_best.pdf"
sbatch --wrap="python src/plot_hist.py -fs training_history_no_neighbor_94.csv -n hist_plots/nn_run.pdf"
sbatch --wrap="python src/plot_hist.py -i 94 -n hist_plots/all.pdf"