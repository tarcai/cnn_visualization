# CNN visualization
/img folder contains the exported images from each convolution layer.

Usefull example: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py


## Script usage:
usage: vgg16_filter_vis.py [-h] [--layer LAYER] [--noimg]

optional arguments:
  -h, --help     show this help message and exit
  --layer LAYER  Name of the layer to visualize. If value is "all", all of the
                 layers will be visualized.
  --noimg        Not save output images