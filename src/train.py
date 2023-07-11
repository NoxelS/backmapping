import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable tensorflow warnings

# from lib.cnn import CNN
import tensorflow as tf
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
from scipy.ndimage import gaussian_filter
import sys
from library.classes.CNN import CNN
from library.classes.TrainingDataGenerator import TrainingDataGenerator

# data_prefix = "/data/users/noel/"      # For smaug
# data_prefix = "/localdisk/noel/"       # For fluffy
data_prefix = "data/"                        # For local

cg_size = (12, 8)
at_size = (138, 8)


cnn = CNN(
    "DOPC",
    keep_checkpoints=True,
    x_train=[],
    y_train=[],
    x_test=[],
    y_test=[],
    path=data_prefix,
    load_path="models/DOPC.h5",
    input_size=cg_size,
    output_size=at_size,
)

train_gen = TrainingDataGenerator(
    input_dir_path=data_prefix + "training",
    output_dir_path=data_prefix + "training",
    input_size=cg_size,
    output_size=at_size,
    shuffle=False,
    batch_size=16
)

test_gen = TrainingDataGenerator(
    input_dir_path=data_prefix + "training",
    output_dir_path=data_prefix + "training",
    input_size=cg_size,
    output_size=at_size,
    shuffle=False,
    batch_size=16
)

cnn.model.fit(
    train_gen,
    batch_size=16,
    epochs=1,
    validation_data=test_gen,
)
