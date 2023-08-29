@echo off

@REM rmdir /s /q data\\tensorboard
@REM rmdir /s /q data\\backup
@REM rmdir /s /q data\\hist

start "Training" cmd /c "python src/train_one_with_neighborhood.py"
start "Tensorboard" cmd /c "tensorboard --logdir=data/tensorboard --host 0.0.0.0 --port 6006"