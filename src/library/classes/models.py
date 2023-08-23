import os
import numpy as np
import tensorflow as tf
from library.classes.losses import BackmappingLoss

class CNN:
    def __init__(
        self,
        input_size,
        output_size,
        display_name,
        data_prefix,
        keep_checkpoints=False,
        load_path=None,
    ):
        super().__init__()

        self.display_name = display_name
        self.keep_checkpoints = keep_checkpoints
        self.load_path = f"{load_path}"
        self.data_prefix = data_prefix

        # Scale the model, this currently only affects the number of filters
        scale = 16

        self.model = tf.keras.Sequential(
            [
                ##### Input layer #####
                tf.keras.layers.Input(input_size, sparse=False),
                ##### Encoder #####
                tf.keras.layers.Conv2D(
                    filters=1 * scale,
                    kernel_size=(2, 2),
                    strides=(1, 1),
                    padding='same',
                    activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                    kernel_initializer=tf.keras.initializers.HeNormal(),
                    use_bias=True,
                    bias_initializer=tf.keras.initializers.HeNormal(),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    padding='same',
                ),
                tf.keras.layers.Conv2D(
                    filters=2 * scale,
                    kernel_size=(2, 2),
                    strides=(1, 1),
                    padding='same',
                    activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=2 * scale,
                    kernel_size=(2, 2),
                    strides=(1, 1),
                    padding='same',
                    activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),
                    strides=(2, 2),
                    padding='same',
                ),
                ##### Latent space #####
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(36, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.BatchNormalization(),
                ##### Decoder #####
                tf.keras.layers.Reshape((6, 6, 1)),
                tf.keras.layers.Conv2DTranspose(
                    filters=2 * scale,
                    kernel_size=(3, 2),
                    strides=(3, 2),
                    padding='same',
                    activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(
                    filters=1 * scale,
                    kernel_size=(3, 2),
                    strides=(3, 2),
                    padding='same',
                    activation=tf.keras.layers.LeakyReLU(alpha=0.01),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(np.prod(output_size), activation='tanh', kernel_initializer=tf.keras.initializers.HeNormal()),
                tf.keras.layers.Reshape(output_size),
                ##### Output #####
            ], name=self.display_name)


        # Compile the model
        self.model.compile(
            optimizer=tf.optimizers.Adam(
                # learning_rate=0.0001,
            ),
            loss=BackmappingLoss(),
            metrics=['accuracy','mae'],
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
            early_stop=False):

        # Create hist dir if it does not exist
        if not os.path.exists(os.path.join(self.data_prefix, "hist")):
            os.makedirs(os.path.join(self.data_prefix, "hist"))

        callbacks = [
            # The BackupAndRestore callback saves checkpoints every save_freq batches,
            # this is set to 1 epoch here to save after each epoch. Make sure to set
            # this to a value greater than one when training on a large dataset, because
            # it can take a long time to save checkpoints.
            tf.keras.callbacks.BackupAndRestore(
                backup_dir=os.path.join(self.data_prefix, "backup", self.display_name),
                save_freq=1,
                delete_checkpoint=not self.keep_checkpoints,
                save_before_preemption=False,
            ),

            # The ReduceLROnPlateau callback monitors a quantity and if no improvement
            # is seen for a 'patience' number of epochs, the learning rate is reduced.
            # Here, it monitors 'val_loss' and if no improvement is seen for 10 epochs,
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.75,
                patience=10,
                verbose=0,
                mode='max',
                min_delta=0.00000005,
                cooldown=5,
                min_lr=0.00000000001,
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
                log_dir=os.path.join(self.data_prefix,'tensorboard'),
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                write_steps_per_second=True,
                update_freq='batch',
                profile_batch=(0, 10),
                embeddings_freq=1,
                embeddings_metadata=None,
            ),
            # Print the min, max and average weight of each layer before each batch.
            # tf.keras.callbacks.LambdaCallback(on_batch_begin=lambda batch, logs: print(self.get_weight_info()))
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
        weights = self.model.get_weights()
        return "\n" + "\n".join([f"[{i}] max_weight={np.max(w)}\tmin_weight={np.min(w)}\tavg_weight={tf.math.reduce_mean(w)}" for i, w in enumerate(weights)])

    def test(self,
             data_generator,
             ):
        """
            Test the model with test data
        """
        X = data_generator.__getitem__(0)[0]
        Y = data_generator.__getitem__(0)[1]
        loss, acc = self.model.evaluate(X, Y, verbose=0)
        print(f"CNN-Test: acc = {100*acc:5.2f}%, loss = {loss:7.4f}")

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
