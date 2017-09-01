#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tarcai
Usefull example: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
"""

import argparse
from scipy.misc import imsave
import numpy as np
from scipy import ndimage
from keras.applications import vgg16
from keras import backend as K

# utility function to normalize a tensor by its L2 norm
def normalize(x):
    return x / (K.sqrt(K.mean(K.square(x))))

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

def get_layer_dict(model):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    return layer_dict

def calculate_gradient(input_img_data, layer, filter_index):
    # step size for gradient ascent
    step = 1.

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer.output
    if layer_name == 'predictions':
        loss = K.mean(model.output[:, filter_index])
    else:
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
    last_loss_value = 0
    for i in range(200):
        if apply_gaussian:
            input_img_data[0] = ndimage.gaussian_filter(input_img_data[0], sigma=1)
        elif apply_uniform:
            input_img_data[0] = ndimage.uniform_filter(input_img_data[0], size=3)
        elif apply_l2decay:
            input_img_data[0] = input_img_data[0]*0.8
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Step %d, loss value:' % i, loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break
        if i > 20 and ((loss_value - last_loss_value) / last_loss_value) < .01 and loss_value > .95:
            break
        last_loss_value = loss_value

    return loss_value

def visualize_filter(filter_index, kept_filters, layer):
    print('Processing filter %d' % filter_index)

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    loss_value = calculate_gradient(input_img_data, layer, filter_index)

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
    imsave('filters_%s%s.png' % (layer_name, img_name_suffix), filters_img)
    print('Filter images are saved')

def get_filter_image_data(number_of_filters):
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
    return kept_filters
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type = str, default = 'block1_conv1', help = 'Name of the layer to visualize. If value is "all_conv", all of the convolution layers will be visualized.')
    parser.add_argument("--noimg", help = 'Not save output images', action = 'store_false')
    parser.add_argument("--gaussian_filter", help = 'Apply gaussian filter on each iteration', action = 'store_true')
    parser.add_argument("--uniform_filter", help = 'Apply uniform filter on each iteration', action = 'store_true')
    parser.add_argument("--l2_decay", help = 'Apply L2 decay on each iteration', action = 'store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_args()
    print(args)
        
    # Set filters
    apply_gaussian = args.gaussian_filter
    apply_uniform = args.uniform_filter
    apply_l2decay = args.l2_decay

    # set image name suffix for filtered running
    img_name_suffix = ''
    if apply_gaussian:
        img_name_suffix = 'gaussian'
    elif apply_uniform:
        img_name_suffix = 'uniform'
    elif apply_l2decay:
        img_name_suffix = 'l2_decay'
    
    # list of the all convolition layers in the CNN
    all_conv_layers = ['block1_conv1', 'block1_conv2',
                       'block2_conv1', 'block2_conv2',
                       'block3_conv1', 'block3_conv2', 'block3_conv3',
                       'block4_conv1', 'block4_conv2', 'block4_conv3',
                       'block5_conv1', 'block5_conv2', 'block5_conv3']
    
    # we will stich the filters on an n x n grid.
    n = 3

    # load VGG16 network with ImageNet weights
    model = vgg16.VGG16(weights='imagenet', include_top=True)
    print('VGG16 model loaded.')    
    layer_dict = get_layer_dict(model)
    all_layers = layer_dict.keys()

    # get the name of the VGG16 layer
    layer_names = []
    if args.layer == 'all_conv':
        layer_names = all_conv_layers
    elif args.layer == 'predictions':
        layer_names = [args.layer]
    else:
        layer_names = [args.layer]
        
    
    # dimensions of the generated pictures for each filter.
    img_width = model.layers[0].input_shape[1]
    img_height = model.layers[0].input_shape[2]
    
    for layer_name in layer_names:
        # get output layar link
        layer = layer_dict[layer_name]
    
        # get the number of filters in the current layer
        number_of_filters = 0
        if layer_name == 'predictions':
            number_of_filters = 1000
        else:
            number_of_filters = layer.get_config()['filters']
        
        # this is the placeholder for the input images
        input_img = model.input

        # get image data
        kept_filters = get_filter_image_data(number_of_filters)

        if args.noimg:        
            # save filters to image
            save_image(kept_filters, img_width, img_height)
            