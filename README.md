# CNN visualization
/img folder contains the exported images from each convolution layer.

Usefull example: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py


## Script usage:

	usage: vgg16_filter_vis.py [-h] [-l LAYER] [--noimg]
    	                       [-r {none,l2,gaussian,uniform}]

	optional arguments:
	  -h, --help            show this help message and exit
	  -l LAYER, --layer LAYER
    	                    Name of the layer to visualize. If value is
        	                "all_conv", all of the convolution layers will be
            	            visualized.
	  --noimg               Not save output images
	  -r {none,l2,gaussian,uniform}, --regularization {none,l2,gaussian,uniform}
    	                    Apply regularization on each iteration