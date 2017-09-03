# CNN visualization
/img folder contains the exported images from each convolution layer.

Usefull example: https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py


## Script usage:

	usage: vgg16_filter_vis.py [-h] [-l LAYER] [--noimg]
	                           [-r {none,l2,gaussian,uniform}] [-n NR_OF_OUTPUTS]
	                           [-L] [-p]

	optional arguments:
	  -h, --help            show this help message and exit
	  -l LAYER, --layer LAYER
    	                    Name of the layer to visualize. If value is "all_conv"
        	                all of the convolution layers will be visualized, if
            	            "all" all of the layers.
	  --noimg               Not save output images
	  -r {none,l2,gaussian,uniform}, --regularization {none,l2,gaussian,uniform}
	                        Apply regularization on each iteration.
	  -n NR_OF_OUTPUTS, --nr_of_outputs NR_OF_OUTPUTS
	                        Number of output nodes to visualize in a layer.
	  -L, --logging         Save losses to file.
	  -p, --plot            Plot losses during gradient iterations.