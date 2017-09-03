#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: tarcai
Usefull example: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
"""

import argparse
from scipy.misc import imsave
import numpy as np
import pandas as pd
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

def calculate_gradient(input_img_data, layer, node_index):
    # step size for gradient ascent
    step = 1.

    # we build a loss function that maximizes the activation
    # of the nth node of the layer considered
    if layer.name in ['predictions', 'fc1', 'fc2']:
        loss = K.mean(layer.output[:, node_index])
    else:
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer.output[:, node_index, :, :])
        else:
            loss = K.mean(layer.output[:, :, :, node_index])
        
    # we compute the gradient of the input picture w.r.t. this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # we run gradient ascent for 20 steps
    last_loss_value = 0
    loss_values = []
    for i in range(200):
        if regularization == 'gaussian':
            input_img_data[0] = ndimage.gaussian_filter(input_img_data[0], sigma=1)
        elif regularization == 'uniform':
            input_img_data[0] = ndimage.uniform_filter(input_img_data[0], size=3)
        elif regularization == 'l2':
            input_img_data[0] = input_img_data[0]*0.8
        loss_value, grads_value = iterate([input_img_data])
        loss_values.append(loss_value)
        input_img_data += grads_value * step

        print('\rStep %d, loss value: %.3f' % (i, loss_value),  end='')
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break
        if i > 20 and ((loss_value - last_loss_value) / last_loss_value) < .01 and loss_value > .95:
            break
        last_loss_value = loss_value
        
    print('')
    return loss_values

def visualize_node(node_index, layer, img_shape, file_name_suffix):
    print('Processing node %d' % node_index)

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 3, img_shape[0], img_shape[1]))
    else:
        input_img_data = np.random.random((1, img_shape[0], img_shape[1], 3))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    loss_values = calculate_gradient(input_img_data, layer, node_index)

    # decode the resulting input image
    if loss_values[-1] > 0:
        img = deprocess_image(input_img_data[0])
        if not args.noimg:        
            save_image(img, node_index, layer.name, file_name_suffix)

    return loss_values

def save_image(img, node_index, layer_name, file_name_suffix):
        # save the result to file
        imsave('%s_%d%s.png' % (layer_name, node_index, file_name_suffix), img)
        print('Image saved')

def visualize_n_nodes(number_of_nodes, layer, img_shape, file_name_suffix, logging):
    # we will only keep the non-zero nodes    
    good_nodes = 0
    for node_index in range(number_of_nodes):
        filename = layer.name + '_' + str(node_index) + file_name_suffix + '.loss.csv'
        loss_values = visualize_node(node_index, layer, img_shape, file_name_suffix)
        if loss_values[-1] > 0:
            good_nodes += 1
            if logging:
                write_losses_to_csv(loss_values, filename)
        else:
            print ('Skip this node.')

        if good_nodes == n:
            break
    
def write_losses_to_csv(losses, filename):
    loss_dataframe = pd.DataFrame(losses, columns=['loss'])
    loss_dataframe.head()
    loss_dataframe.to_csv(filename, index=True, index_label='iteration')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", 
                        "--layer", 
                        type = str, 
                        default = 'block1_conv1', 
                        help = '''Name of the layer to visualize. If value is 
                                  "all_conv" all of the convolution layers will
                                  be visualized, if "all" all of the layers.''')
    parser.add_argument("--noimg", 
                        action = 'store_true',
                        help = 'Not save output images')
    parser.add_argument("-r", 
                        "--regularization", 
                        default = 'none', 
                        choices=['none', 'l2', 'gaussian', 'uniform'],  
                        help = 'Apply regularization on each iteration.')
    parser.add_argument("-n", 
                        "--nr_of_outputs",
                        type = int,
                        default = 1, 
                        help = 'Number of output nodes to visualize in a layer.')
    parser.add_argument("-L", "--logging", 
                        action = 'store_true',
                        help = 'Save losses to file.')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = get_args()
    print(args)
        
    # Set regularization
    regularization = args.regularization

    # set image name suffix for regularization
    file_name_suffix = ''
    if regularization != 'none':
        file_name_suffix = '_' + args.regularization
            
    # list of the all convolition layers in the CNN
    all_conv_layers = ['block1_conv1', 'block1_conv2',
                       'block2_conv1', 'block2_conv2',
                       'block3_conv1', 'block3_conv2', 'block3_conv3',
                       'block4_conv1', 'block4_conv2', 'block4_conv3',
                       'block5_conv1', 'block5_conv2', 'block5_conv3']
    all_layers = all_conv_layers + ['fc1', 'fc2', 'predictions']
    
    # we will visualize and save n images.
    n = args.nr_of_outputs

    # logging
    logging = args.logging

    # load VGG16 network with ImageNet weights
    model = vgg16.VGG16(weights='imagenet', include_top=True)
    print('VGG16 model loaded.')    
    layer_dict = get_layer_dict(model)
    all_layers = layer_dict.keys()

    # get the name of the VGG16 layer
    layer_names = []
    if args.layer == 'all_conv':
        layer_names = all_conv_layers
    elif args.layer == 'all':
        layer_names = all_layers
    else:
        layer_names = [args.layer]
        
    
    # dimensions of the generated pictures.
    img_shape = model.layers[0].input_shape[1:3]
    
    for layer_name in layer_names:
        # get output layar link
        layer = layer_dict[layer_name]
    
        # get the number of nodes in the current layer
        number_of_nodes = 0
        if layer_name in ['predictions', 'fc1', 'fc2']:
            number_of_nodes = layer.units
        else:
            number_of_nodes = layer.filters
        
        # this is the placeholder for the input images
        input_img = model.input

        # get image data
        visualize_n_nodes(number_of_nodes, layer, img_shape, file_name_suffix, logging)
            