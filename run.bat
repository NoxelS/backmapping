@echo off

start "Training" cmd /c "python src/train.py > output.log"
start "Tensorboard" cmd /c "tensorboard --logdir=data/tensorboard --host 0.0.0.0 --port 6006"