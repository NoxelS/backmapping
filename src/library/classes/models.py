import logging
import os
import socket
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from django import conf

from library.classes.generators import (BaseDataGenerator,
                                        inverse_scale_output_ic,
                                        scale_output_ic)
from library.classes.losses import CustomLoss
from library.config import Keys, config
from library.datagen.topology import (get_ic_from_index, get_ic_type,
                                      ic_to_hlabel,
                                      load_extended_topology_info)
from library.notify import send_notification


class IDOFNet:

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
        socket=None,
        host_ip_address=None,
        port=None,
        ic_index=None,
    ):
        """
        This is the base class for all IDOFNets. It contains the basic structure of the IDOFNet and the fit function.

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
            ic_index (int, optional): The index of the internal coordinate. Defaults to None.
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

        # For tracking predictions along training. This is a dict with epoch as key and the predictions as value
        self.predictions = {}

        self.ic_index = ic_index
        if self.ic_index is not None:
            self.ic = get_ic_from_index(ic_index)
            self.ic_type = get_ic_type(self.ic)

        # Create the model based on the specified model type and input/output size/display_name
        self.model = self.model_factory(input_size, output_size, display_name)

        # Compile the model
        self.model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=config(Keys.INITIAL_LEARNING_RATE)),
            loss=self.loss,
            metrics=["mae", "mse"],
            run_eagerly=True,
        )

        # Print the model summary
        self.summary()

        # Load weights if path is given
        if load_path is not None:
            try:
                # Load the model as whole
                self.model = tf.keras.models.load_model(
                    load_path,
                    custom_objects={
                        "CustomLoss": CustomLoss,
                        "custom_loss": CustomLoss,
                        "LeakyReLU": tf.keras.layers.LeakyReLU(alpha=0.01),
                    },
                )
                logging.info("Loaded model successfully from " + load_path)
            except FileNotFoundError:
                logging.info(f"Could not find model at {load_path}, starting from scratch...")
            except Exception as e:
                logging.info("Could not load model from " + load_path + ": " + str(e))
                try:
                    logging.debug("Trying to load model without compiling...")
                    self.model = tf.keras.models.load_model(
                        load_path,
                        custom_objects={
                            "CustomLoss": self.loss,
                            "LeakyReLU": tf.keras.layers.LeakyReLU(alpha=0.01),
                        },
                        compile=False,
                    )
                except Exception as e:
                    logging.debug(e)
                    logging.debug(f"Could not load model from {load_path}, starting from scratch...")
        else:
            logging.debug(f"No backup found at {load_path}, starting from scratch...")

    def model_factory(self, input_size, output_size, display_name):
        """
        This is the model factory for the default model.

        Args:
            input_size: The size of the input. Should be (x, y, 1)
            output_size: The size of the output. Should be (x, y, 1)
            display_name: The name of the model. Used for displaying the model summary and saving checkpoints/logs.

        Returns:
            tf.keras.Sequential: The model
        """

        conv_scale = 4
        return tf.keras.Sequential(
            [
                ##### Input layer #####
                tf.keras.layers.Input(input_size, sparse=False),
                ##### Encoder #####
                tf.keras.layers.Conv2D(
                    filters=2**1 * conv_scale,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="valid",
                    activation=tf.keras.layers.LeakyReLU(alpha=0.03),
                ),
                tf.keras.layers.Conv2D(
                    filters=2**2 * conv_scale,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    activation=tf.keras.layers.LeakyReLU(alpha=0.03),
                ),
                tf.keras.layers.Conv2D(
                    filters=2**4 * conv_scale,
                    kernel_size=(3, 4),
                    strides=(1, 1),
                    padding="valid",
                    activation=tf.keras.layers.LeakyReLU(alpha=0.03),
                ),
                tf.keras.layers.Conv2D(
                    filters=2**5 * conv_scale,
                    kernel_size=(3, 4),
                    strides=(1, 1),
                    padding="valid",
                    activation=tf.keras.layers.LeakyReLU(alpha=0.03),
                ),
                tf.keras.layers.Conv2D(
                    filters=2**6 * conv_scale,
                    kernel_size=(3, 4),
                    strides=(1, 1),
                    padding="valid",
                    activation=tf.keras.layers.LeakyReLU(alpha=0.03),
                ),
                tf.keras.layers.Conv2D(
                    filters=2**7 * conv_scale,
                    kernel_size=(3, 4),
                    strides=(1, 1),
                    padding="valid",
                    activation=tf.keras.layers.LeakyReLU(alpha=0.03),
                ),
                tf.keras.layers.Conv2D(
                    filters=2**8 * conv_scale,
                    kernel_size=(3, 5),
                    strides=(1, 1),
                    padding="valid",
                    activation=tf.keras.layers.LeakyReLU(alpha=0.03),
                ),
                # tf.keras.layers.BatchNormalization(),
                # tf.keras.layers.Conv2D(
                #     filters=2**9 * 8,
                #     kernel_size=(1, 4),
                #     strides=(1, 1),
                #     padding="valid",
                #     activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                # ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(
                    pool_size=(3, 3),
                    padding="same",
                ),
                tf.keras.layers.BatchNormalization(),
                ##### Output #####
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.10),  # Maybe move this after the dense
                tf.keras.layers.Dense(
                    np.prod(output_size),
                    activation="sigmoid",
                    kernel_initializer=tf.keras.initializers.Zeros(),
                    # kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
                ),
                tf.keras.layers.Reshape(output_size),
            ],
            name=f"{display_name}_IDOFNet_default_v_1_0",
        )

    def fit(
        self,
        train_generator,
        validation_gen,
        epochs=150,
        batch_size=64,
        verbose=1,
        early_stop=False,
        use_tensorboard=False,
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
            use_tensorboard (bool, optional): If true tensorboard will be used. Defaults to False.
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
                monitor=config(Keys.LR_SCHEDULER_MONITOR),
                factor=config(Keys.LR_SCHEDULER_FACTOR),
                patience=config(Keys.LR_SCHEDULER_PATIENCE),
                verbose=0,
                mode=config(Keys.LR_SCHEDULER_MODE),
                min_delta=config(Keys.LR_SCHEDULER_MIN_DELTA),
                cooldown=config(Keys.LR_SCHEDULER_COOLDOWN),
                min_lr=config(Keys.LR_SCHEDULER_MIN_LR),
            ),
            # The CSVLogger callback streams epoch results to a CSV file.
            # to monitor the training progress.
            tf.keras.callbacks.CSVLogger(
                os.path.join(self.data_prefix, "hist", f"training_history_{self.display_name}.csv"),
                separator=",",
                append=True,
            ),
            # Update the current epoch and in the future other stuff
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.update_internals(epoch, logs)),
            # Callback to save weights after each epoch
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.save()),
            # Callback to send data to the socket after each epoch
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.send_data_to_socket(epoch, logs)),
            # Callback to log the physical loss after each epoch
            tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: self.log_phys_loss(epoch, logs)),
            # Callback to plot the output histogram after each epoch
            self.plot_output_histogram_callback(train_generator),  # Here we can also use validation_gen, depends on what we want to track
        ]

        if config(Keys.USE_NTFY):
            # Send notifications after n epochs, where n is set via config
            callbacks.append(self.send_notifications())

        if use_tensorboard:
            # The TensorBoard callback writes a log for TensorBoard, which allows
            # you to visualize dynamic graphs of your training and test metrics,
            # as well as activation histograms for the different layers in your model.
            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(self.data_prefix, "tensorboard", self.display_name),
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq="batch",
                ),
            )

            # Add custom metrics to tensorboard
            callbacks.append(
                tf.keras.callbacks.LambdaCallback(on_batch_end=lambda batch, logs: self.custom_tensorboard_metrics(batch, logs)),
            )

        if early_stop:
            # The EarlyStopping callback is used to exit the training early if
            # the validation loss does not improve after 20 epochs.
            # This makes sure that the model does not overfit the training data.
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=0.0001,
                    patience=config(Keys.EARLY_STOP_PATIENCE),
                    verbose=0,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=True,
                )
            )

        # Check history for last epoch
        if os.path.exists(os.path.join(self.data_prefix, "hist", f"training_history_{self.display_name}.csv")):
            last_epoch = 0
            with open(os.path.join(self.data_prefix, "hist", f"training_history_{self.display_name}.csv"), "r") as f:
                last_epoch = len(f.readlines()) - 2  # -2 because header and last empty line
            self.current_epoch = np.max([last_epoch + 1, 0])  # We assume that the last epoch was finished
            logging.debug(f"Starting from epoch {self.current_epoch}")

        # Here we use generators to generate batches of data for the training loop.
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

    def send_data_to_socket(self, epoch, logs):
        """
        Sends the data to the socket
        """
        # Send data to socket
        if self.socket is not None:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((self.host_ip_address, self.port))
                self.socket.send(f"{self.display_name}:{epoch}:{logs['loss']:.6f}".encode())
                self.socket.close()
            except Exception as e:
                logging.error("Could not send data to socket")
                logging.error(e)

    def log_phys_loss(self, epoch, logs):
        """
        Logs the loss in physical units
        """
        mae = 0
        try:
            mae = logs["mae"]
        except KeyError:
            try:
                mae = logs["mean_absolute_error"]
            except KeyError:
                return

        # Get scale
        scaled_mae = inverse_scale_output_ic(self.ic_index, mae) - inverse_scale_output_ic(self.ic_index, 0)
        dim = "Å" if self.ic_type == "bond" else "°"

        print(f" - pmae: {scaled_mae:.6f}{dim}")

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
        summary_dir1 = os.path.join(self.data_prefix, "tensorboard", self.display_name, "custom")
        summary_writer1 = tf.summary.create_file_writer(summary_dir1)

        # Set step
        step = batch + 148 * batch

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

        # Save the model as whole
        self.model.save(path, overwrite=True, save_format="h5")

    def summary(self):
        """
        Prints the model summary
        """
        logging.debug(f"Summary of {self.display_name}")
        self.model.summary()

    def plot_output_histogram(self, data_generator: BaseDataGenerator):
        analysis_folder = os.path.join(config(Keys.DATA_PATH), "analysis", self.display_name)
        file_name = f"predictions_{self.current_epoch}.png"
        save_file_path = os.path.join(analysis_folder, file_name)

        # For debugging
        start_time = time.time()

        # Create directory if it does not exist
        if not os.path.exists(analysis_folder):
            os.makedirs(analysis_folder)

        true_ics = []
        pred_ics = []

        # Predict all data
        for i in range(len(data_generator)):
            x, y_true = data_generator.__getitem__(i)
            y_pred = self.model.predict(x)

            # Remove dimensionality
            y_pred = y_pred[:, 0, 0, 0]
            y_true = y_true[:, 0, 0, 0]

            # Add to list
            for i in range(len(y_true)):
                true_ics.append(y_true[i])
                pred_ics.append(y_pred[i])

        # To numpy
        true_ics = np.array(true_ics)
        pred_ics = np.array(pred_ics)

        # Reverse apply the scale function
        true_ics = inverse_scale_output_ic(self.ic_index, true_ics)
        pred_ics = inverse_scale_output_ic(self.ic_index, pred_ics)

        # Calculate mean and std
        true_mean = np.mean(true_ics)
        pred_mean = np.mean(pred_ics)
        true_std = np.std(true_ics)
        pred_std = np.std(pred_ics)

        # Print Mean
        logging.debug(f"Prediction comparison after {self.current_epoch} epochs: true mean: {true_mean}, pred mean: {pred_mean}")
        logging.debug(f"Prediction comparison after {self.current_epoch} epochs: true std: {true_std}, pred std: {pred_std}")

        # Save the predictions
        self.predictions[self.current_epoch] = pred_ics

        # Loop through all predictions and plot them in one histogram
        plt.figure(figsize=(10, 6))

        # Get min and max epoch
        min_epoch = min(self.predictions.keys())
        max_epoch = max(self.predictions.keys())

        # Calculate alphas based on epoch
        alphas = np.linspace(0.1, 0.9, max_epoch - min_epoch + 1) if max_epoch - min_epoch > 0 else [1]
        first_pred = self.predictions[min_epoch]

        # Loop through all predictions
        for i, pred_ics in enumerate(self.predictions.values()):
            # Make histogram with the same bins
            bins = np.linspace(min(true_ics.min(), pred_ics.min()), max(true_ics.max(), pred_ics.max()), 400)
            # Plot the results as relative histogram
            plt.hist(pred_ics, bins=bins, alpha=alphas[i], color="green")

        # Plot the true values (always the same)
        bins = np.linspace(min(true_ics.min(), first_pred.min()), max(true_ics.max(), first_pred.max()), 400)
        plt.hist(true_ics, bins=bins, alpha=0.9, color="purple", label="True")

        # Get dim of ic
        dim = "Å" if self.ic_type == "bond" else "°"
        xlabel = "Bond length" if self.ic_type == "bond" else "Angle"
        ic_label = ic_to_hlabel(self.ic)

        # Set theme
        plt.style.use("seaborn-paper")

        # Label the plot
        plt.xlabel(f"{xlabel} ({dim})")
        plt.ylabel("Frequency")
        plt.title(f"{xlabel} {ic_label} After {self.current_epoch} Epochs\n(Training Data)")
        plt.grid(True)
        plt.legend(loc="upper right")

        # Save the plot
        plt.savefig(save_file_path)
        plt.close()
        logging.debug(f"Saved plot to {save_file_path}")

        # Log time it took
        logging.debug(f"Time it took to plot output histogram: {time.time() - start_time:.2f} seconds")

    def plot_output_histogram_callback(self, data_generator: BaseDataGenerator):
        """
        Plots the output histogram after 5 epochs
        """
        n = config(Keys.PREDICTION_COOLDOWN)

        def plot_every_N_epochs(epoch, logs):
            if epoch % n == 0:
                try:
                    self.plot_output_histogram(data_generator)
                except Exception as e:
                    logging.error(f"Could not plot output histogram: {e}")

        return tf.keras.callbacks.LambdaCallback(on_epoch_end=plot_every_N_epochs)

    def send_notifications(self) -> tf.keras.callbacks.Callback:
        """
        Send notifications via ntfy after each epoch

        Returns:
            tf.keras.callbacks.Callback: The callback
        """
        epoch_cooldown = config(Keys.NTFY_TRAINING_COOLDOWN)

        def send(epoch, logs):
            if epoch % epoch_cooldown == 0:
                ok = send_notification(
                    title=f"{self.display_name}: Epoch {epoch} finished",
                    message=f"Loss: {logs['loss']:.6f}",
                    tags="info",
                )
                if ok:
                    logging.debug(f"Sent notification for epoch {epoch}.")
                else:
                    logging.error(f"Could not send notification for epoch {epoch}.")

        return tf.keras.callbacks.LambdaCallback(on_epoch_end=send)


class IDOFNet_Reduced(IDOFNet):

    def model_factory(self, input_size, output_size, display_name):
        """
        This is the model factory for the default model.

        Args:
            input_size: The size of the input. Should be (x, y, 1)
            output_size: The size of the output. Should be (x, y, 1)
            display_name: The name of the model. Used for displaying the model summary and saving checkpoints/logs.

        Returns:
            tf.keras.Sequential: The model
        """

        # Get the mean of the std to predict from the extended topology
        mean = self.ic["mean"]
        std = self.ic["std"]

        # Scale the mean to the output, this will be the starting point of the model
        mean_scaled = scale_output_ic(self.ic_index, mean)
        std_scaled = scale_output_ic(self.ic_index, std) - scale_output_ic(self.ic_index, 0)

        conv_scale = 4
        return tf.keras.Sequential(
            [
                ##### Input layer #####
                tf.keras.layers.Input(input_size, sparse=False),
                ##### Encoder #####
                tf.keras.layers.Conv2D(
                    filters=2**1 * conv_scale,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding="valid",
                    activation=tf.keras.layers.LeakyReLU(alpha=0.03),
                ),
                tf.keras.layers.Conv2D(
                    filters=2**2 * conv_scale,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding="same",
                    activation=tf.keras.layers.LeakyReLU(alpha=0.03),
                ),
                tf.keras.layers.Conv2D(
                    filters=2**4 * conv_scale,
                    kernel_size=(3, 4),
                    strides=(1, 1),
                    padding="valid",
                    activation=tf.keras.layers.LeakyReLU(alpha=0.03),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.MaxPool2D(
                    pool_size=(3, 3),
                    padding="same",
                ),
                tf.keras.layers.BatchNormalization(),
                ##### Output #####
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.10),  # Maybe move this after the dense
                tf.keras.layers.Dense(
                    100 * np.prod(output_size),
                    activation="sigmoid",
                    kernel_initializer=tf.keras.initializers.Zeros(),
                    # kernel_initializer=tf.keras.initializers.RandomNormal(mean=mean_scaled, stddev=std_scaled),
                ),
                tf.keras.layers.Dense(
                    np.prod(output_size),
                    activation=config(Keys.OUTPUT_ACTIVATION_FUNCTION),
                    kernel_initializer=tf.keras.initializers.Zeros(),
                    # kernel_initializer=tf.keras.initializers.RandomNormal(mean=mean_scaled, stddev=std_scaled),
                ),
                tf.keras.layers.Reshape(output_size),
            ],
            name=f"{display_name}_IDOFNet_reduced_v_1_0",
        )
