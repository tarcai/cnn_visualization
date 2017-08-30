#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:24:25 2017

@author: tarcai
Usefull example: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
"""

import argparse
from scipy.misc import imsave
import numpy as np
from keras.applications import vgg16
from keras import backend as K

# utility function to normalize a tensor by its L2 norm
def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    return layer_dict[layer_name]


def gradient_step(input_img_data, layer):
    # step size for gradient ascent
    step = 1.

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer.output
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture w.r.t. this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Step %d, loss value:' % i, loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break
    return loss_value

def visualize_filter(filter_index, kept_filters, layer):
    print('Processing filter %d' % filter_index)

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    loss_value = gradient_step(input_img_data, layer)

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))

    return loss_value

def save_image(kept_filters, img_width, img_height):
    # build a black picture with enough space for
    # our n x n filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    filters_img = np.zeros((width, height, 3))
    
    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            filters_img[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img
    
    # save the result to disk
    imsave('filters_%s.png' % (layer_name), filters_img)
    print('Filter images are saved')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type = str, default = 'block1_conv1', help = 'Name of the layer to visualize.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_args()
    print(args)
    
    # get the name of the VGG16 layer
    layer_name = args.layer

    # dimensions of the generated pictures for each filter.
    img_width = 128
    img_height = 128

    # we will stich the filters on an n x n grid.
    n = 3

    # load VGG16 network with ImageNet weights
    model = vgg16.VGG16(weights='imagenet', include_top=False)
    print('VGG16 model loaded.')
    
    # get output layar link
    layer = get_output_layer(model, layer_name)
    
    # get the number of filters in the current layer
    number_of_filters = layer.get_config()['filters']
        
    # this is the placeholder for the input images
    input_img = model.input

    # we will only keep the non-zero filters    
    kept_filters = []
    good_filters = 0
    for filter_index in range(number_of_filters):
        loss_value = visualize_filter(filter_index, kept_filters, layer)
        if loss_value > 0:
            good_filters += 1
        else:
            print ('Skip this filter.')
        if good_filters == (n*n):
            break
        
    # save filters to image
    save_image(kept_filters, img_width, img_height)
           