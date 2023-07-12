import os
import numpy as np
import tensorflow as tf


class CNN:
    def __init__(
        self,
        display_name,
        keep_checkpoints=False,
        x_train=None,
        y_train=None,
        x_test=None,
        y_test=None,
        path=None,
        load_path=None,
        input_size=(12, 8),  # [12 beads] x [3 dimensions + 5 beay types (one hot)]
        output_size=(138, 8),      # [138 atoms] x [3 dimensions + 5 atom types (one hot)]
    ):
        super().__init__()
        self.display_name = display_name
        self.keep_checkpoints = keep_checkpoints
        self.path = path
        self.load_path = f"{path}/{load_path}"

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        # Scale the model
        scale = 1

        self.model = tf.keras.Sequential(
            [
                # Input layer
                tf.keras.layers.Input(input_size, sparse=False),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Reshape((input_size[0], input_size[1], 1)),
                # Encode with convolutions
                tf.keras.layers.Conv2D(64 * scale, (2, 2), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128 * scale, (2, 2), activation='relu'),
                # Latent space
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256 * scale, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Reshape((4, 4, 16)),
                # Decode with transpose convolutions
                tf.keras.layers.Conv2DTranspose(128 * scale, (2, 2), activation='relu'),
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Conv2DTranspose(64 * scale, (2, 2), activation='relu'),
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.15),
                # Output layer
                tf.keras.layers.Dense(np.prod(output_size), activation='sigmoid'),
                tf.keras.layers.Reshape(output_size),
            ], name=self.display_name)

        self.model.summary()

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='mse',
            metrics=['accuracy'],
        )

        # Load weights if path is given
        if load_path is not None and os.path.exists(load_path):
            self.model.load_weights(load_path)
            print("Loaded weights from " + load_path)

        # Checkpoints to save trained network weights
        self.checkpoint_path = f"{path}/backup/"

    # TODO: improve
    def fit(self,
            train_generator,
            test_generator,
            epochs=150, batch_size=64, verbose=1, early_stop=False):

        callbacks = [
            tf.keras.callbacks.BackupAndRestore(
                backup_dir=self.checkpoint_path + self.display_name,
                save_freq='epoch',
                delete_checkpoint=not self.keep_checkpoints,
                save_before_preemption=False,
            ),
            # PlotTrainingHistory(),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.75,
                patience=8,
                verbose=0,
                mode='max',
                min_delta=0.0005,
                cooldown=0,
                min_lr=0.000001,
            ),
            tf.keras.callbacks.CSVLogger(
                f"{self.path}/training_history_{self.display_name}.csv",
                separator=',',
                append=True,
            ),
        ]

        # if early_stop:
        #     callbacks.append(
        #         tf.keras.callbacks.EarlyStopping(
        #             monitor='val_loss',
        #             min_delta=0.0001,
        #             patience=20,
        #             verbose=0,
        #             mode='auto',
        #             baseline=None,
        #             restore_best_weights=True,
        #         ))

        self.hist = self.model.fit(
            x=train_generator,
            # batch_size=batch_size,
            # shuffle=True,
            # self.datagen.flow(
            #     self.x_train,
            #     self.y_train,
            #     batch_size=batch_size,
            #     shuffle=True,
            # ),
            validation_data=test_generator,
            epochs=epochs,
            verbose=verbose,
            # callbacks=callbacks,
        )

        # self.model.trai

    def test(self):
        """
            Test the model with test data
        """
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"CNN-Test: acc = {100*acc:5.2f}%")

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
