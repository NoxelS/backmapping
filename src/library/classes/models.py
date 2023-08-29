import os
import numpy as np
import tensorflow as tf
from library.classes.losses import BackmappingRelativeVectorLoss
from library.classes.generators import PADDING_X, PADDING_Y

import time
import matplotlib.pyplot as plt


class CNN:
    def __init__(
        self,
        input_size,
        output_size,
        display_name,
        data_prefix,
        keep_checkpoints=False,
        load_path=None,
        loss=tf.keras.losses.MeanSquaredError(),
        test_sample=None,
    ):
        super().__init__()

        self.display_name = display_name
        self.keep_checkpoints = keep_checkpoints
        self.load_path = f"{load_path}"
        self.data_prefix = data_prefix
        self.loss = loss
        self.test_sample = test_sample
        self.current_epoch = 0
    
        # For showing samples
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Scale the model, this currently only affects the number of filters
        scale = 32          # Scaled the filter dimensions by this factor
        latent_size = 4**2  # Needs to be a square number

        conv_activation =  tf.keras.layers.LeakyReLU(alpha=0.01)

        self.model = tf.keras.Sequential(
            [
                ##### Input layer #####
                tf.keras.layers.Input(input_size, sparse=False),
                ##### Encoder #####
                tf.keras.layers.Conv2D(
                    filters=2**0 * scale,
                    kernel_size=(2, 2),
                    strides=(1, 1),
                    padding='valid',
                    activation=conv_activation,
                ),
                tf.keras.layers.Conv2D(
                    filters=2**1 * scale,
                    kernel_size=(2, 2),
                    strides=(1, 1),
                    padding='valid',
                    activation=conv_activation,
                ),
                tf.keras.layers.Conv2D(
                    filters=2**2 * scale,
                    kernel_size=(2, 1),
                    strides=(1, 1),
                    padding='valid',
                    activation=conv_activation,
                ),
                tf.keras.layers.Conv2D(
                    filters=2**3 * scale,
                    kernel_size=(2, 1),
                    strides=(1, 1),
                    padding='valid',
                    activation=conv_activation,
                ),
                tf.keras.layers.Conv2D(
                    filters=2**4 * scale,
                    kernel_size=(2, 1),
                    strides=(2, 1),
                    padding='valid',
                    activation=conv_activation,
                ),
                tf.keras.layers.Conv2D(
                    filters=2**5 * scale,
                    kernel_size=(2, 1),
                    strides=(2, 1),
                    padding='valid',
                    activation=conv_activation,
                ),
                tf.keras.layers.Conv2D(
                    filters=2**6 * scale,
                    kernel_size=(2, 1),
                    strides=(2, 1),
                    padding='valid',
                    activation=conv_activation,
                ),
                tf.keras.layers.Conv2D(
                    filters=2**7 * scale,
                    kernel_size=(2, 1),
                    strides=(2, 1),
                    padding='valid',
                    activation=conv_activation,
                ),
                tf.keras.layers.MaxPool2D(
                    pool_size=(3, 3),
                ),
                # tf.keras.layers.Conv2D(
                #     filters=2**8 * scale,
                #     kernel_size=(2, 1),
                #     strides=(1, 1),
                #     padding='valid',
                #     activation=conv_activation,
                # ),
                tf.keras.layers.BatchNormalization(),
                ##### Latent space #####
                # tf.keras.layers.Flatten(),
                # tf.keras.layers.Dense(latent_size, activation='tanh'),
                # tf.keras.layers.Dropout(0.2),
                # tf.keras.layers.BatchNormalization(),
                # ##### Decoder #####
                # tf.keras.layers.Reshape((int(np.floor(np.sqrt(latent_size))), int(np.floor(np.sqrt(latent_size))), 1)),
                # tf.keras.layers.Conv2DTranspose(
                #     filters=2**6 * scale,
                #     kernel_size=(3, 2),
                #     strides=(1, 1),
                #     padding='same',
                #     activation=conv_activation,
                # ),
                # tf.keras.layers.Conv2DTranspose(
                #     filters=2**5 * scale,
                #     kernel_size=(3, 2),
                #     strides=(3, 2),
                #     padding='same',
                #     activation=conv_activation,
                # ),
                # tf.keras.layers.Conv2DTranspose(
                #     filters=2**4 * scale,
                #     kernel_size=(3, 2),
                #     strides=(3, 2),
                #     padding='same',
                #     activation=conv_activation,
                # ),
                # tf.keras.layers.BatchNormalization(),
                ##### Output #####
                tf.keras.layers.Flatten(),
                # tf.keras.layers.Dense(np.prod(output_size) // 2, activation='tanh', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),
                tf.keras.layers.Dense(np.prod(output_size), activation='tanh', kernel_initializer=tf.keras.initializers.Zeros()), #tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)),
                tf.keras.layers.Reshape(output_size)
            ], name=self.display_name)

        # Compile the model
        self.model.compile(
            optimizer=tf.optimizers.Adam(
                # learning_rate=0.00001, # Start with 1% of the default learning rate
            ),
            loss=self.loss,
            metrics=['accuracy', 'mae'],
            run_eagerly=True
        )

        self.model.summary()

        # Load weights if path is given
        if load_path is not None and os.path.exists(load_path):
            self.model.load_weights(load_path)
            print("Loaded weights from " + load_path)

    def fit(self,
            train_generator,
            validation_gen,
            epochs=150,
            batch_size=64,
            verbose=1,
            early_stop=False,
            ):

        # Create hist dir if it does not exist
        if not os.path.exists(os.path.join(self.data_prefix, "hist")):
            os.makedirs(os.path.join(self.data_prefix, "hist"))

        callbacks = [
            # The BackupAndRestore callback saves checkpoints every save_freq batches,
            # this is set to 1 epoch here to save after each epoch. Make sure to set
            # this to a value greater than one when training on a large dataset, because
            # it can take a long time to save checkpoints.
            tf.keras.callbacks.experimental.BackupAndRestore(
                backup_dir=os.path.join(self.data_prefix, "backup", self.display_name),
                # save_freq=1,
                # delete_checkpoint=not self.keep_checkpoints,
                # save_before_preemption=False,
            ),

            # The ReduceLROnPlateau callback monitors a quantity and if no improvement
            # is seen for a 'patience' number of epochs, the learning rate is reduced.
            # Here, it monitors 'val_loss' and if no improvement is seen for 10 epochs,
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.75,
                patience=3,
                verbose=0,
                mode='max',
                min_delta=0.00000005,
                cooldown=5,
                min_lr=0.0000000000001,
            ),

            # The CSVLogger callback streams epoch results to a CSV file.
            # #TODO Send the CSV file via email or telegram after each epoch
            # to monitor the training progress.
            tf.keras.callbacks.CSVLogger(
                os.path.join(self.data_prefix, "hist", f"training_history_{self.display_name}.csv"),
                separator=',',
                append=True,
            ),

            # The TensorBoard callback writes a log for TensorBoard, which allows
            # you to visualize dynamic graphs of your training and test metrics,
            # as well as activation histograms for the different layers in your model.
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.data_prefix, 'tensorboard', self.display_name),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='batch',
            ),
            # Update the current epoch and in the future other stuff
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.update_internals(epoch, logs)),
            # Print the min, max and average weight of each layer before each batch.
            tf.keras.callbacks.LambdaCallback(on_batch_begin=lambda batch, logs: self.get_weight_info()),
            # Track the test sample after each epoch
            # tf.keras.callbacks.LambdaCallback(on_batch_end=lambda epoch, logs: self.track_test_samle(epoch, logs)),
            # Track the test sample after each batch (live)
            # tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: self.live_track_test_samle(batch, logs)),
            # Add custom metrics to tensorboard
            tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: self.custom_tensorboard_metrics(batch, logs)),
        ]

        if early_stop:
            # The EarlyStopping callback is used to exit the training early if
            # the validation loss does not improve after 20 epochs.
            # This makes sure that the model does not overfit the training data.
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.0001,
                    patience=20,
                    verbose=0,
                    mode='auto',
                    baseline=None,
                    restore_best_weights=True,
                ))

        # Open plot for live tracking TODO: fix this
        # self.fig.show()
        # self.fig.canvas.draw()

        # Here we use generators to generate batches of data for the training.
        # The training data is currently generated by the RelativeVectorsTrainingDataGenerator
        # and is the main thing that changes the input data structure to the desired output.
        self.hist = self.model.fit(
            train_generator,
            validation_data=validation_gen,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
        )

    def get_weight_info(self):
        # Write thouse metrics to tensorboard

        now = time.localtime()
        subdir = time.strftime("%d-%b-%Y_%H.%M.%S", now)

        summary_dir1 = os.path.join("data", "tensorboard", self.display_name, "custom")
        summary_writer1 = tf.summary.create_file_writer(summary_dir1)

        # TODO: FIX THIS
        # # Loop over layers
        # for i, layer in enumerate(self.model.layers):
        #     name = layer.name
        #     weights = np.array(layer.get_weights())
        #     print(weights.min())

        #     # Find min, max and avg of the kernel weights
        #     min_weight = weights.min().min()
        #     max_weight = weights.max().min()
        #     avg_weight = weights.mean().mean()

        #     # Write scalars
        #     with summary_writer1.as_default():
        #         tf.summary.scalar(f"min_weight_{name}", min_weight, step=i)
        #         tf.summary.scalar(f"max_weight_{name}", max_weight, step=i)
        #         tf.summary.scalar(f"avg_weight_{name}", avg_weight, step=i)
        summary_writer1.flush()

        weights = self.model.get_weights()
        # return "\n" + "\n".join([f"[{i}] max_weight={np.max(w)} - min_weight={np.min(w)} - avg_weight={tf.math.reduce_mean(w)}" for i, w in enumerate(weights)])
        return ""

    def track_test_samle(self, batch, logs=None):
        # This function runs after every epochs and predicts one output sample and builds images for a video
        # Additionally the molecule will be shown and the plot will be updated

        # Predict the output
        pred = self.model.predict(self.test_sample[0][0:1, :, :, :])[0, :, :, 0]
        true = self.test_sample[1][0, :, :, 0]
        
        # Remove padding
        pred = pred[PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y]
        true = true[PADDING_X:-PADDING_X, PADDING_Y:-PADDING_Y]

        # Make 3D scatter
        pred_positions = []
        true_positions = []
        for i, bond in enumerate(pred):
            pred_positions.append(pred[i, :])
        for i, bond in enumerate(true):
            true_positions.append(true[i, :])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter([x[0] for x in true_positions], [x[1] for x in true_positions], [x[2] for x in true_positions], c='b', marker='o')
        ax.scatter([x[0] for x in pred_positions], [x[1] for x in pred_positions], [x[2] for x in pred_positions], c='r', marker='o')

        # Draw lines between the atoms predicted and true positions
        for i in range(len(true_positions)):
            ax.plot([true_positions[i][0], pred_positions[i][0]], [true_positions[i][1], pred_positions[i][1]], [true_positions[i][2], pred_positions[i][2]], color='gray', linestyle='-', linewidth=0.1)

        # Add legend
        ax.legend(["True", "Predicted"])

        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_title(f"Atom Positions Epoch {self.current_epoch:03d} Batch {batch:03d}")

        # Fix min, max of every coordinate of the true positions
        min_x = min([x[0] for x in true_positions])
        max_x = max([x[0] for x in true_positions])
        min_y = min([x[1] for x in true_positions])
        max_y = max([x[1] for x in true_positions])
        min_z = min([x[2] for x in true_positions])
        max_z = max([x[2] for x in true_positions])

        # Make size fixed with padding
        padding_factor = 0.1
        scale_factor = 1
        fixed_padding = 0.1
        
        ax.set_xlim(scale_factor * min_x - padding_factor * (max_x - min_x) - fixed_padding, scale_factor * max_x + padding_factor * (max_x - min_x) + fixed_padding)
        ax.set_ylim(scale_factor * min_y - padding_factor * (max_y - min_y) - fixed_padding, scale_factor * max_y + padding_factor * (max_y - min_y) + fixed_padding)
        ax.set_zlim(scale_factor * min_z - padding_factor * (max_z - min_z) - fixed_padding, scale_factor * max_z + padding_factor * (max_z - min_z) + fixed_padding)

        # Make angle fixed from 45 in x and 45 in y
        ax.view_init(azim=-40, elev=35)

        # Create directory if it does not exist
        if not os.path.exists(os.path.join(self.data_prefix, "hist")):
            os.makedirs(os.path.join(self.data_prefix, "hist"))

        if not os.path.exists(os.path.join(self.data_prefix, "hist", "pred")):
            os.makedirs(os.path.join(self.data_prefix, "hist", "pred"))
            
        # Find the max number of the files in the folder
        index = len(os.listdir(os.path.join(self.data_prefix, "hist", "pred")))

        # Save the figure
        # TODO: Fix save folder for atom specific models
        fig.savefig(os.path.join(self.data_prefix, "hist", "pred", f"batch_{index}.png"))

    def live_track_test_samle(self, batch, logs=None):
        # Does the same as track_test_samle but does not save the figure and shows it in a window and updates it
        # after each batch

        # Predict the output
        pred = self.model.predict(self.test_sample[0][0:1, :, :, :])[0, :, :, 0]
        true = self.test_sample[1][0, :, :, 0]

        # Make 3D scatter
        pred_positions = []
        true_positions = []
        for i, bond in enumerate(pred):
            pred_positions.append(pred[i, :])
        for i, bond in enumerate(true):
            true_positions.append(true[i, :])

        self.ax.clear()
        self.ax.scatter([x[0] for x in true_positions], [x[1] for x in true_positions], [x[2] for x in true_positions], c='b', marker='o')
        self.ax.scatter([x[0] for x in pred_positions], [x[1] for x in pred_positions], [x[2] for x in pred_positions], c='r', marker='o')

        # Add legend
        self.ax.legend(["True", "Predicted"])

        # Add labels
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        self.ax.set_title(f"Atom Positions Batch {batch}")

        # Fix min, max of every coordinate of the true positions
        min_x = min([x[0] for x in true_positions])
        max_x = max([x[0] for x in true_positions])
        min_y = min([x[1] for x in true_positions])
        max_y = max([x[1] for x in true_positions])
        min_z = min([x[2] for x in true_positions])
        max_z = max([x[2] for x in true_positions])

        # Make size fixed with padding
        padding_factor = 0
        self.ax.set_xlim(min_x - padding_factor * (max_x - min_x), max_x + padding_factor * (max_x - min_x))
        self.ax.set_ylim(min_y - padding_factor * (max_y - min_y), max_y + padding_factor * (max_y - min_y))
        self.ax.set_zlim(min_z - padding_factor * (max_z - min_z), max_z + padding_factor * (max_z - min_z))

        # Make angle fixed from 45 in x and 45 in y
        self.ax.view_init(azim=-45, elev=35)

        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_internals(self, epoch, logs):
        self.current_epoch = epoch
        
    def custom_tensorboard_metrics(self, batch, logs):
        # Add custom metrics to tensorboard for each batch
        
        # Write loss to tensorboard        
        summary_dir1 = os.path.join("data", "tensorboard", self.display_name, "custom")
        summary_writer1 = tf.summary.create_file_writer(summary_dir1)
        
        # Set step
        step = batch + 148 * batch # TODO: Fix this
        
        with summary_writer1.as_default():
            tf.summary.scalar("loss_b", logs["loss"], step=step)
            tf.summary.scalar("accuracy_b", logs["accuracy"], step=step)
            tf.summary.scalar("mae_b", logs["mae"], step=step)

    def test(self, data_generator):
        """
            Test the model with test data
        """
        X = data_generator.__getitem__(0)[0]
        Y = data_generator.__getitem__(0)[1]
        loss, acc = self.model.evaluate(X, Y, verbose=0)
        print(f"CNN-Test: acc = {100*acc:5.2f}%, loss = {loss:7.4f}")

    def find_dataset_with_lowest_loss(self, data_generator):
        """
            Finds the dataset with the lowest loss
        """
        lowest_loss = 100000
        lowest_loss_dataset = None
        for i in range(data_generator.__len__()):
            X = data_generator.__getitem__(i)[0]
            Y = data_generator.__getitem__(i)[1]
            loss, acc = self.model.evaluate(X, Y, verbose=0)
            if loss < lowest_loss:
                lowest_loss = loss
                lowest_loss_dataset = i
        return lowest_loss_dataset

    def predict(self, x):
        """
            Predicts the output for a given input
        """
        return self.model.predict(x)

    def activation(self, x, layer_name):
        """
            Returns the activation map of a given input for a given layer
        """
        for layer in self.model.layers:
            if layer.name == layer_name:
                return layer.call(x)
            else:
                x = layer.call(x)

    def save(self, path=None):
        """
            Saves the model
        """
        path = self.load_path if path is None else path
        self.model.save(path)
