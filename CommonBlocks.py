import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, Conv2D,  AveragePooling2D, Conv3D, Activation, \
    AveragePooling3D, Add, UpSampling2D, BatchNormalization, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model


class DBlock(tf.keras.layers.Layer):

    def __init__(self, filters, dimReduction=True, firstRelu=True):
        super(DBlock, self).__init__()
        self.filters = filters
        self.dimReduction = dimReduction  # for discriminators
        self.firstRelu = firstRelu  # for discriminators

        # 1x1 conv
        self.conv1_1x1 = Conv2D(filters, kernel_size=(1, 1), activation=None, padding='same')
        # pooling
        if dimReduction:
            self.pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

        # activation
        if firstRelu:
            self.activation1 = Activation('relu')
        # 3x3 conv
        self.conv1_3x3 = Conv2D(filters, kernel_size=(3, 3), padding='same')
        # activation
        self.activation2 = Activation('relu')
        # 3x3 conv
        self.conv2_3x3 = Conv2D(filters, kernel_size=(3, 3), padding='same')
        # pooling
        if dimReduction:
            self.pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')  # mean/sum pooling

        # sum
        self.add1 = Add()

    def call(self, inputs):

        # upper path
        fx = self.conv1_1x1(inputs)
        if self.dimReduction:
            fx = self.pool1(fx)

        # lower path
        if self.firstRelu:  # false for beginning of discriminators
            gx = self.activation1(inputs)
        else:
            gx = inputs
        gx = self.conv1_3x3(gx)
        gx = self.activation2(gx)
        gx = self.conv2_3x3(gx)
        if self.dimReduction:  # false for end of discriminators
            gx = self.pool2(gx)

        out = self.add1([fx, gx])

        return out


class DBlock3D(tf.keras.layers.Layer):

    def __init__(self, filters, dimReduction=True, firstRelu=True):
        super(DBlock3D, self).__init__()
        self.filters = filters
        self.dimReduction = dimReduction  # for discriminators
        self.firstRelu = firstRelu  # for discriminators

        # 1x1 conv + spectral norm
        self.conv1_1x1 = tfa.layers.SpectralNormalization(Conv3D(filters, kernel_size=(1, 1, 1), activation=None, padding='same'))
        # pooling
        if dimReduction:  # remove padding for temporal discriminator
            self.pool1 = AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))

        # activation
        if firstRelu:
            self.activation1 = Activation('relu')
        # 3x3 conv + spectral norm
        self.conv1_3x3 = tfa.layers.SpectralNormalization(Conv3D(filters, kernel_size=(3, 3, 3), padding='same'))
        # activation
        self.activation2 = Activation('relu')
        # 3x3 conv + spectral norm
        self.conv2_3x3 = tfa.layers.SpectralNormalization(Conv3D(filters, kernel_size=(3, 3, 3), padding='same'))
        # pooling
        if dimReduction:  # remove padding for temporal discriminator (5 instead of 6 results)
            self.pool2 = AveragePooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))  # mean/sum pooling

        # sum
        self.add1 = Add()

    def call(self, inputs):

        # upper path
        fx = self.conv1_1x1(inputs)
        if self.dimReduction:
            fx = self.pool1(fx)

        # lower path
        if self.firstRelu:  # false for beginning of discriminators
            gx = self.activation1(inputs)
        else:
            gx = inputs
        gx = self.conv1_3x3(gx)
        gx = self.activation2(gx)
        gx = self.conv2_3x3(gx)
        if self.dimReduction:  # false for end of discriminators
            gx = self.pool2(gx)

        out = self.add1([fx, gx])

        return out


class GBlock(tf.keras.layers.Layer):

    def __init__(self, filters, dimIncrease=True):
        super(GBlock, self).__init__()
        self.filters = filters
        self.dimIncrease = dimIncrease  # for sampler

        # upsampling with nearest neighbour interpolation
        if dimIncrease:
            self.pool1 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='nearest')
        # 1x1 conv:  spectral norm must be added?
        self.conv1_1x1 = Conv2DTranspose(filters, kernel_size=(1, 1), activation=None, padding='same')

        # normalization
        self.norm1 = BatchNormalization()
        # activation
        self.activation1 = Activation('relu')
        if dimIncrease:
            self.pool2 = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='nearest')
        # 3x3 conv:  spectral norm must be added?
        self.conv1_3x3 = Conv2DTranspose(filters, kernel_size=(3, 3), padding='same')
        # normalization
        self.norm2 = BatchNormalization()
        # activation
        self.activation2 = Activation('relu')
        # 3x3 conv:  spectral norm must be added?
        # introduces an unreasonable amount of parameters for some reason
        self.conv2_3x3 = Conv2DTranspose(filters, kernel_size=(3, 3), padding='same')
        # sum
        self.add1 = Add()

    def call(self, inputs):

        # upper path
        if self.dimIncrease:
            fx = self.pool1(inputs)
        else:
            fx = inputs
        fx = self.conv1_1x1(fx)

        # lower path
        gx = self.norm1(inputs)
        gx = self.activation1(gx)
        if self.dimIncrease:
            gx = self.pool2(gx)
        gx = self.conv1_3x3(gx)
        gx = self.norm2(gx)
        gx = self.activation2(gx)
        gx = self.conv2_3x3(gx)

        out = self.add1([fx, gx])

        return out


class LBlock(tf.keras.layers.Layer):  # transposed convolutions?

    def __init__(self, filters, filters_in):
        super(LBlock, self).__init__()
        self.filters = filters

        # 1x1 conv
        self.conv1_1x1 = Conv2D((filters - filters_in), kernel_size=(1, 1), activation=None, padding='same')
        # concat
        self.concat1 = Concatenate()

        # activation
        self.activation1 = Activation('relu')
        # 3x3 conv
        self.conv1_3x3 = Conv2D(filters, kernel_size=(3, 3), padding='same')
        # activation (redundant object but better readable)
        self.activation2 = Activation('relu')
        # 3x3 conv
        self.conv2_3x3 = Conv2D(filters, kernel_size=(3, 3), padding='same')
        # sum
        self.add1 = Add()

    def call(self, inputs):

        # upper path
        fx = self.conv1_1x1(inputs)
        fx = self.concat1([inputs, fx])

        # lower path
        gx = self.activation1(inputs)
        gx = self.conv1_3x3(gx)
        gx = self.activation2(gx)
        gx = self.conv2_3x3(gx)

        out = self.add1([fx, gx])

        return out
