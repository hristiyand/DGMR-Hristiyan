# pip install tfrecord

import tfrecord
from matplotlib import pyplot as plt
from PIL import Image

# data description
context_description = {"radar": "byte", "sample_prob": "float", "osgb_extent_top": "int", "osgb_extent_left": "int",
                       "osgb_extent_right": "int", "osgb_extent_bottom": "int", "end_time_timestamp": "int"}
sequence_description = {}

# load data sequence
loader = tfrecord.tfrecord_loader("/content/drive/MyDrive/radar_data_test.tfrecord",
                                  None,
                                  context_description,
                                  sequence_description=sequence_description)


test = []
for context, sequence_description in loader:

    test = context["radar"].reshape((6144,512))
    plt.figure(figsize = (20,20))
    plt.imshow(test[:256][0:256],aspect='auto')
    #print(context["sample_prob"])
    #print(context["end_time_timestamp"])
    #print(context["osgb_extent_left"])  #random 256x256 crops
    #print(context["osgb_extent_right"]) #256
    #print(context["osgb_extent_top"])
    #print(context["osgb_extent_bottom"]) #256
