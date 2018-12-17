# neural-style-tf

Final project repo for CS230, forked from original repo.
Originally based on techniques described in the papers: 
* [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
* [Artistic style transfer for videos](https://arxiv.org/abs/1604.08610)
by Manuel Ruder, Alexey Dosovitskiy, Thomas Brox
* [Preserving Color in Neural Artistic Style Transfer](https://arxiv.org/abs/1606.05897)
by Leon A. Gatys, Matthias Bethge, Aaron Hertzmann, Eli Shechtman  


## Setup
#### Dependencies:
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [opencv](http://opencv.org/downloads.html)
* Download the [VGG-19 model weights](http://www.vlfeat.org/matconvnet/pretrained/) (see the "VGG-VD models from the *Very Deep Convolutional Networks for Large-Scale Visual Recognition* project" section). 

## Overview
### Mask.py
Mask.py generates segmentation masks for images. These masks are then written to data/masks/.
We then pass in the path to the content image and path to the style image, as well as the name of both of the 
masks for the content and style image (presumably generated from Mask.py). 

### neural_style.py
Assuming we've generated the required masks and the style/content images are located accordingly, we can generate masked, styled images using neural-style.py like so:


```
python neural_style.py --content_img data/tiny_raw/ikea/ikea_chair.jpeg \
						--style_imgs data/tiny_raw/etc/west_elm.jpg \
						--style_imgs_dir . --content_img_dir . 
						--max_size 256  --max_iterations 1000 \
						--print_iterations 10 --device /cpu:0 --verbose \
						--style_mask --style_mask_imgs ikea_chair_mask.png \
						--content_mask --content_mask_img west_elm_mask.png \
						--img_name dual_mask_ikea_westelm \
						--style_layer_group 2 \
```

This would style ikea_chair.jpeg in the style of west_elm.jpg, trimming the images to 256 pixels and using 100 iterations. Additionally, it uses a mask specified by --style_mask_imgs to decide where on the content image to place the style from the style image, specified by ikea_chair_mask.png here. Next, it uses a content_mask to determine where to take style from the style image, specified by west_elm_mask.png here. (Apologies for the coutner-intuitive names.) Finally, it uses style_layer_group 2, which specifies which layers to use in the calculation of style loss. Valid options being 1, 2, and 3. 1 uses a variety of layers across the network, 1 uses two earlier layers, and 3 uses two deeper layers. It then outputs the resulting image, along with the jobs metadata, to a dual_mask_ikea_westelm directory.

## Acknowledgements

The implementation is based on the projects: 
* Torch (Lua) implementation 'neural-style' by [jcjohnson](https://github.com/jcjohnson)
* Torch (Lua) implementation 'artistic-videos' by [manuelruder](https://github.com/manuelruder)
