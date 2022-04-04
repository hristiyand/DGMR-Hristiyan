import random
import tensorflow as tf
import tensorflow_addons as tfa
from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import BatchNormalization, Flatten, Input, Activation, AveragePooling2D, Add, Dense
from tensorflow.keras.models import Model
from CommonBlocks import DBlock, DBlock3D


class TemporalDiscriminator(tf.keras.layers.Layer):

    def __init__(self, filters_out):
        super(TemporalDiscriminator, self).__init__()
        self.filters_out = filters_out

        # 3D Blocks with spectrally normalized convolutions
        self.dblock1 = DBlock3D(self.filters_out / 16, firstRelu=False)
        self.dblock2 = DBlock3D(self.filters_out / 8)
        # D Blocks
        self.dblock3 = DBlock(self.filters_out / 4)
        self.dblock4 = DBlock(self.filters_out / 2)
        self.dblock5 = DBlock(self.filters_out)
        # preserves W,H,C
        self.dblock6 = DBlock(self.filters_out, dimReduction=False)

        # last size reduction (sum-pooling - mean pooling) CRLE
        self.pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        # spectrally normalized linear layer
        self.flat1 = Flatten()
        self.linear = tfa.layers.SpectralNormalization(Dense(768))
        # sum 8 samples
        self.add1 = Add()
        # batch norm? should come after the linear layer to normalize it
        # two normalizations?
        self.norm = BatchNormalization()
        self.output_layer = Dense(1)
        self.activation = Activation('relu')  # after sum for binary classification

    def call(self, input_images):
        # random crop (same crop for the whole sequence to examine temporal consistency)
        # tf.image.random_crop random crop can be used as well
        crop_center_x = random.randint(64, 191)
        crop_center_y = random.randint(64, 191)
        crop = input_images[:, :, (crop_center_x - 64):(crop_center_x + 64), (crop_center_y - 64):(crop_center_y + 64),
               :]
        # define space-to-depth as list for every sample
        s2d = []

        # crop and stack
        for i in range(0, crop.shape[1]):
            temp = tf.nn.space_to_depth(crop[:, i, :, :, :], 2)
            s2d.append(temp)
            # if i == 0:
            #    s2d = temp
            # else:
            #    s2d = tf.stack([s2d, temp], axis=1)
        # problems with None dimension, stack manually
        s2d = tf.stack(
            [s2d[0], s2d[1], s2d[2], s2d[3], s2d[4], s2d[5], s2d[6], s2d[7], s2d[8], s2d[9], s2d[10], s2d[11],
             s2d[12], s2d[13], s2d[14], s2d[15], s2d[16], s2d[17], s2d[18], s2d[19], s2d[20], s2d[21]], axis=1)

        #s2d = s2d[:, 0, :, :, :, :]
        #s2d = tf.transpose(s2d, perm=[2, 1, 3, 4, 5])

        x = self.dblock1(s2d)
        x = self.dblock2(x)
        # results in length 6 instead of 5 with padding='same'

        out = []
        # separately process 5 resulting representations
        for current in range(0,x.shape[1]):

            x_temp = self.dblock3(x[:,current,:,:,:])
            x_temp = self.dblock4(x_temp)
            x_temp = self.dblock5(x_temp)
            x_temp = self.dblock6(x_temp)
            x_temp = self.pool2(x_temp)
            # flatten
            x_temp = self.flat1(x_temp)
            x_temp = self.linear(x_temp)
            out.append(x_temp)
        # add layers up
        x = self.add1([out[0], out[1], out[2], out[3], out[4]])
        x = self.norm(x)
        output = self.output_layer(x)
        output = self.activation(output)

        return output


class SpatialDiscriminator(tf.keras.layers.Layer):  # transposed convolutions?

    def __init__(self, filters_out):
        super(SpatialDiscriminator, self).__init__()
        self.filters_out = filters_out

        # initial size reduction
        self.pool1 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        # D Blocks
        self.dblock1 = DBlock(self.filters_out / 16, firstRelu=False)
        self.dblock2 = DBlock(self.filters_out / 8)
        self.dblock3 = DBlock(self.filters_out / 4)
        self.dblock4 = DBlock(self.filters_out / 2)
        self.dblock5 = DBlock(self.filters_out)
        # preserves W,H,C
        self.dblock6 = DBlock(self.filters_out, dimReduction=False)

        # last size reduction (sum-pooling = mean pooling)
        self.pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')
        # spectrally normalized linear layer
        self.flat1 = Flatten()
        self.linear = tfa.layers.SpectralNormalization(Dense(768))
        # sum 8 samples
        self.add1 = Add()
        # batch norm after the linear layer to normalize it
        self.norm = BatchNormalization()
        self.output_layer = Dense(1)
        self.activation = Activation('relu')  # after sum for binary classification

    def call(self, inputs): # __call__ to plot summary and scheme

        # take 8 out of 18 random samples
        temp_list = [*range(0, 18)]
        image_indizes = random.sample(temp_list, 8)

        out = []

        # iterate through drawn samples
        for current in image_indizes:
            # pooling
            x = inputs[:, current, :, :, :]
            x = self.pool1(x)
            # s2d
            x = tf.nn.space_to_depth(x, 2)
            x = self.dblock1(x)
            x = self.dblock2(x)
            x = self.dblock3(x)
            x = self.dblock4(x)
            x = self.dblock5(x)
            x = self.dblock6(x)
            x = self.pool2(x)
            x = self.flat1(x)
            x = self.linear(x)
            out.append(x)

        x = self.add1([out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]])
        x = self.norm(x)
        output = self.output_layer(x)
        output = self.activation(output)

        return output

# for testing

#inputs = Input(shape=(22, 256, 256, 1), name="test")
#xx = SpatialDiscriminator(768)(inputs)
#xx = TemporalDiscriminator(768)(inputs)
#outputs = xx

#model = Model(inputs, outputs)
#model.summary()
#plot_model(model, to_file='D:/Desktop/temp_disc.png', show_shapes=True, show_layer_names=True)
