from os import name

import tensorflow as tf


class PolarAngleLayer(tf.keras.layers.Layer):
    """
    This is a custom layer that takes an input tensor and applies a series of dense layers with different activation functions to it. The output of the layer is the concatenation of the outputs of the dense layers.
    This is inspired from https://stackoverflow.com/questions/76120369/tensorflow-keras-multiple-activation-functions-in-one-layer
    """

    def __init__(self, units: int, activations: list = ["cos", "sin"], **kwargs):
        self.units = units
        self.activations = activations
        num_fns = len(self.activations)
        remainder = units % num_fns
        adj_units = units - remainder
        units_per_fn = adj_units / num_fns
        self.activations_data = {activations[-1]: {"Units": units_per_fn + remainder}}

        for fn in self.activations[:-1]:
            self.activations_data[fn] = {"Units": units_per_fn}

        super(PolarAngleLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        for activation, data in self.activations_data.items():
            dense = tf.keras.layers.Dense(units=data["Units"], activation=activation)
            print("Building layer with activation: ", "Dense_" + activation)
            dense.build(input_shape)
            self.activations_data[activation]["Dense"] = dense

    def call(self, input_data):
        output_data_list = []

        for data in self.activations_data.values():
            output = data["Dense"].call(input_data)
            output_data_list.append(output)

        output_data = tf.concat(output_data_list, axis=1)
        return output_data

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activations": self.activations,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
