import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, Concatenate
from tensorflow.keras.models import Model
from CommonBlocks import DBlock
from keras.utils.vis_utils import plot_model
import tensorflow_addons as tfa


class ConditioningStack(tf.keras.layers.Layer):

    def __init__(self, filters_output, context_images=4, down_layers=4):
        super(ConditioningStack, self).__init__()
        self.filters_output = filters_output
        self.context_images = context_images
        self.down_layers = down_layers

        # different D Blocks to separate processing, can be optimized
        self.block11 = DBlock(self.filters_output / 8 / 4)
        self.block12 = DBlock(self.filters_output / 8 / 4)
        self.block13 = DBlock(self.filters_output / 8 / 4)
        self.block14 = DBlock(self.filters_output / 8 / 4)

        self.block21 = DBlock(self.filters_output / 4 / 4)
        self.block22 = DBlock(self.filters_output / 4 / 4)
        self.block23 = DBlock(self.filters_output / 4 / 4)
        self.block24 = DBlock(self.filters_output / 4 / 4)

        self.block31 = DBlock(self.filters_output / 2 / 4)
        self.block32 = DBlock(self.filters_output / 2 / 4)
        self.block33 = DBlock(self.filters_output / 2 / 4)
        self.block34 = DBlock(self.filters_output / 2 / 4)

        self.block41 = DBlock(self.filters_output / 1 / 4)
        self.block42 = DBlock(self.filters_output / 1 / 4)
        self.block43 = DBlock(self.filters_output / 1 / 4)
        self.block44 = DBlock(self.filters_output / 1 / 4)

        # 3x3 convolutions with spectral normalization
        self.conv1_3x3 = tfa.layers.SpectralNormalization(
            Conv2D(filters_output / 16, kernel_size=(3, 3), padding='same'))
        self.conv2_3x3 = tfa.layers.SpectralNormalization(
            Conv2D(filters_output / 8, kernel_size=(3, 3), padding='same'))
        self.conv3_3x3 = tfa.layers.SpectralNormalization(
            Conv2D(filters_output / 4, kernel_size=(3, 3), padding='same'))
        self.conv4_3x3 = tfa.layers.SpectralNormalization(
            Conv2D(filters_output / 2, kernel_size=(3, 3), padding='same'))

        # concat
        self.concat1 = Concatenate()
        # activations
        self.activation1 = Activation('relu')
        self.activation2 = Activation('relu')
        self.activation3 = Activation('relu')
        self.activation4 = Activation('relu')

    def call(self, input_images):  # __ __ to call method for debugging

        x1 = []
        x2 = []
        x3 = []
        x4 = []
        layers = []

        for layerNum in range(self.down_layers):

            if layerNum == 0:

                # space-to-depth
                context1 = tf.nn.space_to_depth(input_images[0], 2)
                context2 = tf.nn.space_to_depth(input_images[1], 2)
                context3 = tf.nn.space_to_depth(input_images[2], 2)
                context4 = tf.nn.space_to_depth(input_images[3], 2)

                x1 = self.block11(context1)
                x2 = self.block12(context2)
                x3 = self.block13(context3)
                x4 = self.block14(context4)

                x_out = self.concat1([x1, x2, x3, x4])
                x_out = self.conv1_3x3(x_out)
                x_out = self.activation1(x_out)

            elif layerNum == 1:
                x1 = self.block21(x1)
                x2 = self.block22(x2)
                x3 = self.block23(x3)
                x4 = self.block24(x4)
                x_out = Concatenate()([x1, x2, x3, x4])
                x_out = self.conv2_3x3(x_out)
                x_out = self.activation2(x_out)

            elif layerNum == 2:
                x1 = self.block31(x1)
                x2 = self.block32(x2)
                x3 = self.block33(x3)
                x4 = self.block34(x4)
                x_out = Concatenate()([x1, x2, x3, x4])
                x_out = self.conv3_3x3(x_out)
                x_out = self.activation3(x_out)

            else:
                x1 = self.block41(x1)
                x2 = self.block42(x2)
                x3 = self.block43(x3)
                x4 = self.block44(x4)
                x_out = Concatenate()([x1, x2, x3, x4])
                x_out = self.conv4_3x3(x_out)
                x_out = self.activation4(x_out)

            layers.append(x_out)

        return [layers[0], layers[1], layers[2], layers[3]]

# for testing

# image1 = Input(shape=(256, 256, 1), name="image1")
# image2 = Input(shape=(256, 256, 1), name="image2")
# image3 = Input(shape=(256, 256, 1), name="image3")
# image4 = Input(shape=(256, 256, 1), name="image4")

# inputs = [image1, image2, image3, image4]
# stack = inputs
# stack = ConditioningStack(768)(stack)
# outputs = stack

# model = Model(inputs, outputs)
# model.summary()
# plot_model(model, to_file='D:/Desktop/model_plot.png', show_shapes=True, show_layer_names=True)
