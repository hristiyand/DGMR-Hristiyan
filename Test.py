from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf
from ConditioningStack import ConditioningStack
from Generator import Generator
from LatentConditioningStack import LatentConditioningStack
from Discriminators import TemporalDiscriminator, SpatialDiscriminator
from keras.utils.vis_utils import plot_model

# proof that space to depth works properly
# e = tf.constant([[[ [1],   [2],  [5],  [6]],
#      [ [3],   [4],  [7],  [8]],
#      [ [9],  [10], [13],  [14]],
#      [ [11], [12], [15],  [16]]]])
# print(e)
# x = tf.nn.space_to_depth(e,2)
# print(x)

# inputs
image1 = Input(shape=(256, 256, 1), name="image1")
image2 = Input(shape=(256, 256, 1), name="image2")
image3 = Input(shape=(256, 256, 1), name="image3")
image4 = Input(shape=(256, 256, 1), name="image4")
normal = Input(shape=(8, 8, 8), name="latent")

images = [image1, image2, image3, image4]
inputs = [image1, image2, image3, image4, normal]

# Conditioning Stack
context = ConditioningStack(768)(images)
# Latent Conditioning Stack
latent = LatentConditioningStack()(normal)
context.append(latent)
# Generator
generator = Generator(768)(context)
images.extend(generator)    #concat context to predictions for temporal discriminator
input_temp_disc = tf.stack(images[:], axis=1)   # stack tensors to an additional dimension
input_spat_disc = tf.stack(generator[:], axis=1)
# Discriminators
temp_disc = TemporalDiscriminator(768)(input_temp_disc)
spat_disc = SpatialDiscriminator(768)(input_spat_disc)

outputs = [generator, temp_disc, spat_disc]

model = Model(inputs, outputs)
model.summary()
#plot_model(model, to_file='D:/Desktop/model_whole.png', show_shapes=True, show_layer_names=True)
