import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization
from tensorflow.keras.models import Model
from CommonBlocks import GBlock
from keras.utils.vis_utils import plot_model


# takes in hidden state and feature vector
class UpscaleData(tf.keras.layers.Layer):  # repeated four times for upscaling to input size

    def __init__(self, filters):
        super(UpscaleData, self).__init__()
        self.filters = filters

        # ToDo: convGRU cell comes here (existing function is ConvLSTM2D)
        self.conv1 = tfa.layers.SpectralNormalization(Conv2D(filters, kernel_size=(1, 1), activation=None, padding='same'))
        self.gblock1 = GBlock(filters, dimIncrease=False)
        # reduces filters and doubles channels
        self.gblock2 = GBlock(filters / 2, dimIncrease=True)

    def call(self, insert):

        # ToDo: must be changed to ConvGRU cell
        hiddenStates = [insert[0]]
        x = self.conv1(insert[1])  # stateGRU instead
        x = self.gblock1(x)
        x = self.gblock2(x)

        hiddenStates.append(x)
        output = hiddenStates

        return output

# operations after the last upscaling
class OutputStack(tf.keras.layers.Layer):  # transposed convolutions?

    def __init__(self, filters):
        super(OutputStack, self).__init__()
        self.filters = filters

        self.norm = BatchNormalization()
        self.activation = Activation('relu')
        # spectral normalization
        self.conv = tfa.layers.SpectralNormalization(Conv2D(filters, kernel_size=(1, 1), activation=None, padding='same'))

    def call(self, insert):

        x = self.norm(insert)
        x = self.activation(x)
        x = self.conv(x)

        # depth-to-space to output 256x256 image
        x = tf.nn.depth_to_space(x, 2)
        result = x
        return result


class Generator(tf.keras.layers.Layer):  # transposed convolutions?

    def __init__(self, filters):
        super(Generator, self).__init__()
        self.filters = filters

        self.upscale1 = UpscaleData(self.filters)
        self.upscale2 = UpscaleData(self.filters/2)
        self.upscale3 = UpscaleData(self.filters/4)
        self.upscale4 = UpscaleData(self.filters/8)
        self.predictions = OutputStack(4)

    def call(self, insert):

        predictions = []
        new_states = []

        for i in range(0, 18):  # result from each timestep is prediciton + four new hidden states
            if i == 0:
                # bottom to top with initial states with z vector
                x1 = UpscaleData(768)([insert[3], insert[4]])
                x2 = UpscaleData(384)([insert[2], x1[1]])
                x3 = UpscaleData(192)([insert[1], x2[1]])
                x4 = UpscaleData(96)([insert[0], x3[1]])
                new_states = [x1[0], x2[0], x3[0], x4[0]]  # new states are iterated top to bottom
                new_output = OutputStack(4)(x4[1])
            else:
                # top to bottom with previous states with same z vector
                x1 = UpscaleData(768)([new_states[0], insert[4]])
                x2 = UpscaleData(384)([new_states[1], x1[1]])
                x3 = UpscaleData(192)([new_states[2], x2[1]])
                x4 = UpscaleData(96)([new_states[3], x3[1]])
                new_states = [x1[0], x2[0], x3[0], x4[0]]
                new_output = OutputStack(4)(x4[1])
            predictions.append(new_output)

        return predictions

# for testing

#context1 = Input(shape=(64, 64, 48), name="first_state")
#context2 = Input(shape=(32, 32, 96), name="second_state")
#context3 = Input(shape=(16, 16, 192), name="third_state")
#context4 = Input(shape=(8, 8, 384), name="fourth_state")

#latent = Input(shape=(8, 8, 768), name="z")
#inputs = [context1, context2, context3, context4, latent]

#x = Generator(768)(inputs)
#outputs = x

#model = Model(inputs, outputs)
#model.summary()
#plot_model(model, to_file='D:/Desktop/generator2.png', show_shapes=True, show_layer_names=True)
