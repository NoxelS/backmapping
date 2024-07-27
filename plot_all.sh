#!/bin/bash

mkdir -p hist_plots

# HP runs
python src/plot_hist.py -lp leaky_relu_alpha -d -c relu_run -n hist_plots/relu_run.pdf
python src/plot_hist.py -lp batch_size -d -c batch_size -n hist_plots/bs_run.pdf
python src/plot_hist.py -lp dropout_rate -d -c do_run -n hist_plots/do_run.pdf
python src/plot_hist.py -lp feature_extraction_units -d -c fe_run -n hist_plots/fe_run.pdf
python src/plot_hist.py -d -c activation_run -n hist_plots/activation_run.pdf
python src/plot_hist.py -lp max_augmentation_angle -d -c aug_run -n hist_plots/aug_run.pdf
python src/plot_hist.py -lp batch_size -d -c bs_test -n hist_plots/bs_run.pdf
python src/plot_hist.py -lp initial_learning_rate -d -c lr_run -n hist_plots/lr_run.pdf
python src/plot_hist.py -lp filters_scale -d -c capacity -n hist_plots/capacity_run.pdf
python src/plot_hist.py -d -c loss_test -n hist_plots/loss_run.pdf
python src/plot_hist.py -e 0 200 -lp initial_learning_rate -d -c lr_run_0.00000 -n hist_plots/lr_run_small.pdf
python src/plot_hist.py -lp initial_learning_rate -d -fs training_history_lr_run_0.000007_94.csv -n hist_plots/lr_run_best.pdf
python src/plot_hist.py -d -fs training_history_no_neighbor_94.csv -n hist_plots/nn_run.pdf

# Prod runs
python src/plot_hist.py -c prod -n hist_plots/bonds_full.pdf -d -e 0 800 --filter-ic-type bond
python src/plot_hist.py -c prod -n hist_plots/bonds_small.pdf -d -s -e 700 800 --filter-ic-type bond
python src/plot_hist.py -c prod -n hist_plots/angles_full.pdf -d -e 0 250 --filter-ic-type angle
python src/plot_hist.py -c prod -n hist_plots/angles_small.pdf -d -s -e 200 250 --filter-ic-type angle
python src/plot_hist.py -c prod -n hist_plots/dihedrals_full.pdf -d -e 0 250 --filter-ic-type dihedral
python src/plot_hist.py -c prod -n hist_plots/dihedrals_small.pdf -d -s -e 200 250 --filter-ic-type dihedral
