from library.classes.losses import BackmappingRelativeVectorLoss
from library.classes.generators import PADDING_X, PADDING_Y
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import socket
import time
import os


class CNN:
    def __init__(
        self,
        input_size: tuple,
        output_size: tuple,
        display_name: str,
        data_prefix: str,
        keep_checkpoints: bool = False,
        load_path: str = None,
        loss: tf.keras.losses.Loss = tf.keras.losses.MeanSquaredError(),
        test_sample: tuple = None,
        socket = None,
        host_ip_address = None,
        port = None
    ):
        """
        This is the base class for all CNNs. It contains the basic structure of the CNN and the fit function.

        Args:
            input_size (tuple): The size of the input. Should be (x, y, 1)
            output_size (tuple): The size of the output. Should be (x, y, 1)
            display_name (str): The name of the model. Used for displaying the model summary and saving checkpoints/logs.
            data_prefix (str): The prefix for all data. Used for saving the model and saving tensorboard logs.
            keep_checkpoints (bool, optional): If true checkpoints will be kept. Defaults to False.
            load_path (str, optional): The path to the weights to load. Defaults to None.
            loss (tf.keras.losses.Loss, optional): The loss function to use. Defaults to tf.keras.losses.MeanSquaredError().
            test_sample (tuple, optional): The test sample to track after each epoch. Defaults to None.
            socket (socket, optional): The socket to send data to. Defaults to None.
            host_ip_address (str, optional): The ip address of the host. Defaults to None.
            port (int, optional): The port to send data to. Defaults to None.
        """
        super().__init__()

        self.display_name = display_name
        self.keep_checkpoints = keep_checkpoints
        self.load_path = load_path
        self.data_prefix = data_prefix
        self.loss = loss
        self.test_sample = test_sample
        self.current_epoch = 0
        self.socket = socket
        self.host_ip_address = host_ip_address
        self.port = port

        # For showing samples
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Scale the model, this currently only affects the number of filters
        scale = 1          # Scaled the filter dimensions by this factor

        # The activation function to use for the convolutional layers
        conv_activation = tf.keras.layers.LeakyReLU(alpha=0.01)

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
                tf.keras.layers.BatchNormalization(),
                ##### Output #####
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(np.prod(output_size), activation='tanh', kernel_initializer=tf.keras.initializers.Zeros()),  # tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1)),
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
        else:
            print("No backup found, starting from scratch...")

    def fit(self,
            train_generator,
            validation_gen,
            epochs=150,
            batch_size=64,
            verbose=1,
            early_stop=False,
            ):
        """
            Trains the model with the given data

            Args:
                train_generator (tf.keras.utils.Sequence): The training data generator
                validation_gen (tf.keras.utils.Sequence): The validation data generator
                epochs (int, optional): The number of epochs to train. Defaults to 150.
                batch_size (int, optional): The batch size. Defaults to 64.
                verbose (int, optional): The verbosity level. Defaults to 1.
                early_stop (bool, optional): If true early stop will be used. Defaults to False.
        """

        # Create hist dir if it does not exist
        if not os.path.exists(os.path.join(self.data_prefix, "hist")):
            os.makedirs(os.path.join(self.data_prefix, "hist"))

        callbacks = [
            # The BackupAndRestore callback saves checkpoints every save_freq batches,
            # this is set to 1 epoch here to save after each epoch. Make sure to set
            # this to a value greater than one when training on a large dataset, because
            # it can take a long time to save checkpoints.
            # tf.keras.callbacks.experimental.BackupAndRestore(
            #     backup_dir=os.path.join(self.data_prefix, "backup", self.display_name),
            #     # save_freq=1,
            #     delete_checkpoint=not self.keep_checkpoints,
            #     # save_before_preemption=False,
            # ),

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

            # Track the test sample after each epoch
            # tf.keras.callbacks.LambdaCallback(on_batch_end=lambda epoch, logs: self.track_test_samle(epoch, logs)),

            # Track the test sample after each batch (live)
            # tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: self.live_track_test_samle(batch, logs)),

            # Add custom metrics to tensorboard
            tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: self.custom_tensorboard_metrics(batch, logs)),

            # Callback to save weights after each epoch
            tf.keras.callbacks.LambdaCallback(on_batch_end=lambda epoch, logs: self.save()),
            
            # Callback to send data to the socket after each epoch
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.send_data_to_socket(epoch, logs)),
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

        # Check history for last epoch
        if os.path.exists(os.path.join(self.data_prefix, "hist", f"training_history_{self.display_name}.csv")):
            with open(os.path.join(self.data_prefix, "hist", f"training_history_{self.display_name}.csv"), "r") as f:
                last_line = f.readlines()[-1]
                self.current_epoch = int(last_line.split(",")[0]) + 1
                print(f"Starting from epoch {self.current_epoch}")

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
            initial_epoch=self.current_epoch,
        )

    def track_test_samle(self, batch, logs=None):
        """
            Tracks the test sample after each batch and saves the figure to the hist folder.

            Args:
                batch (int): The current batch
                logs (dict, optional): The logs from the training. Defaults to None.
        """

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
        """
        Does the same as track_test_samle but live. This is not working yet.

        Args:
            batch (int): The current batch
            logs (dict, optional): The logs from the training. Defaults to None.
        """

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
        
    def send_data_to_socket(self, epoch, logs):
        """
            Sends the data to the socket
        """
        # Send data to socket
        if self.socket is not None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host_ip_address, self.port))
            self.socket.send(f"{self.display_name}:{epoch}:{logs['loss']:.6f}".encode())
            self.socket.close()

    def update_internals(self, epoch, logs):
        """
        Update the current epoch and in the future other stuff like learning rate etc.

        Args:
            epoch (int): The current epoch
            logs (dict): Dict with the logs from the training.
        """
        self.current_epoch = epoch

    def custom_tensorboard_metrics(self, batch, logs):
        """
            Add custom metrics to tensorboard for each batch. This is not working correctly yet.

            Args:
                batch (int): The current batch
                logs (dict): Dict with the logs from the training.
        """
        # Add custom metrics to tensorboard for each batch

        # Write loss to tensorboard
        summary_dir1 = os.path.join("data", "tensorboard", self.display_name, "custom")
        summary_writer1 = tf.summary.create_file_writer(summary_dir1)

        # Set step
        step = batch + 148 * batch  # TODO: Fix this

        with summary_writer1.as_default():
            tf.summary.scalar("loss_b", logs["loss"], step=step)
            tf.summary.scalar("accuracy_b", logs["accuracy"], step=step)
            tf.summary.scalar("mae_b", logs["mae"], step=step)

    def test(self, data_generator):
        """
            Test the model with a test data generator. This only uses the first batch of the generator.
        """
        X = data_generator.__getitem__(0)[0]
        Y = data_generator.__getitem__(0)[1]
        return self.model.evaluate(X, Y, verbose=0, return_dict=True)

    def find_dataset_with_lowest_loss(self, data_generator):
        """
            Finds the dataset with the lowest loss
        """
        lowest_loss = 100000
        lowest_loss_dataset = None
        for i in range(data_generator.__len__()):
            X = data_generator.__getitem__(i)[0]
            Y = data_generator.__getitem__(i)[1]
            res = self.model.evaluate(X, Y, verbose=0, return_dict=True)
            loss, acc = res["loss"], res["accuracy"]
            if loss < lowest_loss:
                lowest_loss = loss
                lowest_loss_dataset = i
        return lowest_loss_dataset, lowest_loss

    def find_dataset_with_highest_loss(self, data_generator):
        """
            Finds the dataset with the highest loss
        """
        highest_loss = 0
        highest_loss_dataset = None
        for i in range(data_generator.__len__()):
            X = data_generator.__getitem__(i)[0]
            Y = data_generator.__getitem__(i)[1]
            res = self.model.evaluate(X, Y, verbose=0, return_dict=True)
            loss, acc = res["loss"], res["accuracy"]
            if loss > highest_loss:
                highest_loss = loss
                highest_loss_dataset = i
        return highest_loss_dataset, highest_loss

    def plot_data_loss_growth(self, data_generator):
        """
            Plot the loss of each dataset in the generator to see if the loss is growing
        """
        # Clear plots
        plt.close("all")

        # Make plot
        fig = plt.figure()
        ax = fig.add_subplot(111)

        losses = []
        for i in range(data_generator.__len__()):
            X = data_generator.__getitem__(i)[0]
            Y = data_generator.__getitem__(i)[1]
            res = self.model.evaluate(X, Y, verbose=0, return_dict=True)
            losses.append(res["loss"])

        ax.plot(losses)
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Loss")
        ax.set_title("Loss of each dataset")

        # Multiply ticks by batch size
        ticks = ax.get_xticks()
        ticks = [int(tick * data_generator.batch_size) for tick in ticks]
        ax.set_xticklabels(ticks)

        # Save plot
        fig.savefig(f"loss_growth_{self.display_name}.png")

        # Show plot
        plt.show()

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

        if not path:
            raise Exception("No path given to save the model")

        self.model.save_weights(path, overwrite=True, save_format="H5")
