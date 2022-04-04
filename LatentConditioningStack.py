import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, Reshape, Multiply
from tensorflow.keras.models import Model
from CommonBlocks import LBlock
from keras.utils.vis_utils import plot_model


class SpatialAttentionModule(tf.keras.layers.Layer):

    def __init__(self, ):
        super(SpatialAttentionModule, self).__init__()

        # constant values matching the paper description
        filters = 192
        map_x = 8
        map_y = 8

        # feature space f
        self.path_f_conv = Conv2D(filters, kernel_size=(1, 1), activation=None, padding='same')
        # reduce dim for matrix multiplication from 3 to 2
        self.path_f2_reshape = Reshape((map_x * map_y, filters), input_shape=(map_x, map_y, filters))

        # feature space g
        self.path_g_conv = Conv2D(filters, kernel_size=(1, 1), activation=None, padding='same')
        # reduce dim for matrix multiplication from 3 to 2
        self.path_g2_reshape = Reshape((filters, map_x * map_y), input_shape=(filters, map_x, map_y))

        self.softmax = Activation('softmax')  # ToDo: must be changed to row wise softmax

        # feature space h
        self.path_h_conv = Conv2D(filters, kernel_size=(1, 1), activation=None, padding='same')
        self.path_h2_reshape = Reshape((map_x * map_y, filters), input_shape=(map_x, map_y, filters))

        # increase dimensions back to 3
        self.reshape1 = Reshape((map_x, map_y, filters), input_shape=(map_x * map_y, filters))
        self.conv1 = Conv2D(192, kernel_size=(1, 1), activation=None, padding='same')  # num of channels

        # initialize trainable gamma to 0
        self.initializer = tf.keras.initializers.Zeros()
        self.gamma = self.initializer(shape=(1, 1, 1))
        self.multiply1 = Multiply()

        self.add1 = Add()  # add learnable parameter gamma

    def call(self, inputs):

        # feature space f
        fx = self.path_f_conv(inputs)
        fx = self.path_f2_reshape(fx)

        # feature space g
        gx = self.path_g_conv(inputs)
        # fx is transposed in SAGAN paper but should deliver the same results (non-local networks paper)
        gx = tf.transpose(gx, perm=(0, 3, 1, 2))
        gx = self.path_g2_reshape(gx)

        # fuse f and g
        first = tf.matmul(fx, gx)
        first = self.softmax(first)

        hx = self.path_h_conv(inputs)
        hx = self.path_h2_reshape(hx)

        # fuse fg and h
        second = tf.matmul(first, hx)
        # dim 2 -> 3
        second = self.reshape1(second)
        second = self.conv1(second)

        gamma = self.gamma
        result = self.multiply1([gamma, second])
        out = self.add1([inputs, result])

        return out


class LatentConditioningStack(tf.keras.layers.Layer):  # transposed convolutions?

    def __init__(self, filters_in=8, filters_out=768, use_attention=True):
        super(LatentConditioningStack, self).__init__()
        # in case we need the filter values
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.use_attention = use_attention

        # ToDo: random values normal distribution are to be inserted as input instead
        # following two options to generate values are presented
        self.normal_draws = tf.random.normal([8, 8, 8], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None,
                                             name="randomDraws")  # initializer works as well?
        self.initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.inputs = self.initializer(shape=(1, 8, 8, 8))

        # 3x3 conv
        self.conv1_3x3 = Conv2D(filters_in, kernel_size=(3, 3), padding='same')
        # LBlocks
        self.lblock1 = LBlock(24, 8)
        self.lblock2 = LBlock(48, 24)
        self.lblock3 = LBlock(192, 48)
        self.lblock4 = LBlock(768, 192)

        # spatial attention
        if self.use_attention:
            self.attention = SpatialAttentionModule()

    def call(self, base):

        latent = self.conv1_3x3(base)
        latent = self.lblock1(latent)
        latent = self.lblock2(latent)
        latent = self.lblock3(latent)
        if self.use_attention:
            latent = self.attention(latent)
        latent = self.lblock4(latent)

        return latent

# for testing

#inputs = Input(shape=(8, 8, 8), name="test")
#x = LatentConditioningStack()(inputs)
#outputs = x

#model = Model(inputs, outputs)
#model.summary()
# cannot plot model with attention due to multiplication (not a layer)
#plot_model(model, to_file='D:/Desktop/latent_stack.png', show_shapes=True, show_layer_names=True)

