echo Starting training
python src/train.py 48
echo Training complete, running plot hist
python src/plot_hist.py
echo Plotting complete, running post analysis
python src/post_analysis_single.py 48
python src/post_analysis_single.py 48