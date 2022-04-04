import tensorflow as tf
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.layers import Conv2D,  AveragePooling2D, Conv3D, Activation, \
    AveragePooling3D, Add, UpSampling2D, BatchNormalization, Conv2DTranspose, Concatenate, \
    Reshape, Multiply


def d_block(x, filters):
    fx = Conv2D(filters, kernel_size = (1,1), activation=None, padding='same')(x)
    fx = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(fx)

    gx = Activation('relu')(x)
    gx = Conv2D(filters, kernel_size = (3,3), padding='same')(gx)
    gx = Activation('relu')(gx)
    gx = Conv2D(filters, kernel_size = (3,3), padding='same')(gx)
    gx = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(gx) #average?

    out = Add()([fx,gx])

    return out

def d_block3D(x, filters):

    fx = SpectralNormalization(Conv3D(filters, kernel_size = (1,1,1), activation=None, padding='same'))(x)
    fx = AveragePooling3D(pool_size=(2, 2, 2), strides=(2,2,2), padding='same')(fx)

    gx = Activation('relu')(x)
    gx = SpectralNormalization(Conv3D(filters, kernel_size = (3,3,3), padding='same'))(gx)
    gx = Activation('relu')(gx)
    gx = SpectralNormalization(Conv3D(filters, kernel_size = (3,3,3), padding='same'))(gx)
    gx = AveragePooling3D(pool_size=(2, 2, 2), strides=(2,2,2), padding='same')(gx) #average?

    out = Add()([fx,gx])

    return out

def d_block3DtemporalDis1(x, filters):

    fx = Conv3D(filters, kernel_size = (1,1,1), activation=None, padding='same', data_format = 'channels_first')(x)
    fx = AveragePooling3D(pool_size=(2, 2, 2), strides=(2,2,2), padding='same',)(fx)

    gx = Conv3D(filters, kernel_size = (3,3,3), padding='same',data_format = 'channels_first')(x)
    gx = Activation('relu')(gx)
    gx = Conv3D(filters, kernel_size = (3,3,3), padding='same',data_format = 'channels_first')(gx)
    gx = AveragePooling3D(pool_size=(2, 2, 2), strides=(2,2,2), padding='same')(gx) #average?

    out = Add()([fx,gx])

    return out

def d_block_spatialDis1(x, filters):
    fx = Conv2D(filters, kernel_size = (1,1), activation=None, padding='same')(x)
    fx = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(fx)

    gx = Conv2D(filters, kernel_size = (3,3), padding='same')(x)
    gx = Activation('relu')(gx)
    gx = Conv2D(filters, kernel_size = (3,3), padding='same')(gx)
    gx = AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='same')(gx) #average?

    out = Add()([fx,gx])

    return out

def d_block_noDownsampling(x, filters):
    fx = Conv2D(filters, kernel_size = (1,1), activation=None, padding='same')(x)

    gx = Activation('relu')(x)
    gx = Conv2D(filters, kernel_size = (3,3), padding='same')(gx)
    gx = Activation('relu')(gx)
    gx = Conv2D(filters, kernel_size = (3,3), padding='same')(gx)

    out = Add()([fx,gx])

    return out

def g_block(x, filters):

    fx = UpSampling2D(size=(2, 2), data_format = 'channels_last', interpolation='nearest')(x)
    fx = Conv2DTranspose(filters, kernel_size = (1,1), activation=None, padding='same')(fx) #correct?

    gx = BatchNormalization()(x)
    gx = Activation('relu')(gx)
    gx = UpSampling2D(size=(2, 2), data_format = 'channels_last', interpolation='nearest')(gx)
    gx = Conv2DTranspose(filters, kernel_size = (3,3), padding='same')(gx)#correct?
    gx = BatchNormalization()(gx)
    gx = Activation('relu')(gx)
    gx = Conv2DTranspose(filters, kernel_size = (3,3), padding='same')(gx)#correct?

    out = Add()([fx,gx])

    return out


def l_block(x, filters):

    filters_in = x.get_shape()[-1]

    fx = Conv2D((filters-filters_in), kernel_size = (1,1), padding='same')(x)
    fx = Concatenate()([x,fx])

    gx = Activation('relu')(x)
    gx = Conv2D(filters, kernel_size = (3,3), padding='same')(gx)
    gx = Activation('relu')(gx)
    gx = Conv2D(filters, kernel_size = (3,3), padding='same')(gx)

    out = Add()([fx,gx])

    return out

def sp_attention_module(x, shape):

    shape = x.get_shape()
    filters = int(shape[-1] /8)
    map_x = shape[-3]
    map_y = shape[-2]

    fx = Conv2D(filters, kernel_size=(1,1), activation=None, padding='same')(x)
    gx = Conv2D(filters, kernel_size=(1,1), activation=None, padding='same')(x)
    gx = tf.transpose(gx, perm=(0,3,1,2))

    fx = Reshape((map_x*map_y,filters), input_shape=(map_x,map_y,filters))(fx)
    gx = Reshape((filters,map_x*map_y), input_shape=(filters,map_x,map_y))(gx)

    first = tf.matmul(fx,gx)
    first = Activation('softmax')(first) #must be changed to row wise softmax

    hx = Conv2D(filters, kernel_size=(1,1), activation=None, padding='same')(x)
    hx = Reshape((map_x*map_y,filters), input_shape=(map_x,map_y,filters))(hx)
    second = tf.matmul(first,hx)

    second = Reshape((map_x,map_y,filters), input_shape=(map_x*map_y,filters))(second)
    second = Conv2D(shape[-1], kernel_size=(1,1), activation=None, padding='same')(second)

    initializer = tf.keras.initializers.Zeros()
    gamma = initializer(shape=(1,1,1))
    o = Multiply()([gamma, second])

    out = Add()([x,o]) ## add learnable parameter gamma


    return out

# manual implementation of space to depth
class space_to_depth(tf.keras.layers.Layer):

  def __init__( self , patch_size ):
    super( space_to_depth , self ).__init__()
    self.patch_size = patch_size

  def call(self, input ):

    patches = []
    print(len(input.shape))
    print(input.shape)

    input_image_size = input.shape[1]
    # split into four patches and concatenate
    for i in range( 0 , input_image_size , self.patch_size ):
        for j in range( 0 , input_image_size , self.patch_size ):
            patches.append( input[ : , i : i + self.patch_size , j : j + self.patch_size , : ] )
    concatted = tf.keras.layers.Concatenate()([patches[0],patches[1],patches[2],patches[3]])

    return concatted
