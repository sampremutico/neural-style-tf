# neural-style-tf

This is a TensorFlow implementation of several techniques described in the papers: 
* [Image Style Transfer Using Convolutional Neural Networks](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)
by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
* [Artistic style transfer for videos](https://arxiv.org/abs/1604.08610)
by Manuel Ruder, Alexey Dosovitskiy, Thomas Brox
* [Preserving Color in Neural Artistic Style Transfer](https://arxiv.org/abs/1606.05897)
by Leon A. Gatys, Matthias Bethge, Aaron Hertzmann, Eli Shechtman  


### Segmentation
Style can be transferred to semantic segmentations in the content image.
<p align="center">
<img src="examples/segmentation/00110.jpg" height="180px">
<img src="examples/segmentation/00110_mask.png" height="180px">
<img src="examples/segmentation/00110_output.png" height="180px">  
<img src="examples/segmentation/00017.jpg" height="180px">
<img src="examples/segmentation/00017_mask.png" height="180px">
<img src="examples/segmentation/00017_output.png" height="180px">  

<img src="examples/segmentation/00768.jpg" height="180px">
<img src="examples/segmentation/00768_mask.png" height="180px">
<img src="examples/segmentation/00768_output.png" height="180px">
<img src="examples/segmentation/02630.png" height="180px">
<img src="examples/segmentation/02630_mask.png" height="180px">
<img src="examples/segmentation/02630_output.png" height="180px"> 
</p>

### Layer Representations
The feature complexities and receptive field sizes increase down the CNN heirarchy.

Here we reproduce Figure 3 from [the original paper](https://arxiv.org/abs/1508.06576):
<table align='center'>
<tr align='center'>
<td></td>
<td>1 x 10^-5</td>
<td>1 x 10^-4</td>
<td>1 x 10^-3</td>
<td>1 x 10^-2</td>
</tr>
<tr>
<td>conv1_1</td>
<td><img src="examples/layers/conv1_1_1e5.png" width="192"></td>
<td><img src="examples/layers/conv1_1_1e4.png" width="192"></td>
<td><img src="examples/layers/conv1_1_1e3.png" width="192"></td>
<td><img src="examples/layers/conv1_1_1e2.png" width="192"></td>
</tr>
<tr>
<td>conv2_1</td>
<td><img src="examples/layers/conv2_1_1e5.png" width="192"></td>
<td><img src="examples/layers/conv2_1_1e4.png" width="192"></td>
<td><img src="examples/layers/conv2_1_1e3.png" width="192"></td>
<td><img src="examples/layers/conv2_1_1e2.png" width="192"></td>
</tr>
<tr>
<td>conv3_1</td>
<td><img src="examples/layers/conv3_1_1e5.png" width="192"></td>
<td><img src="examples/layers/conv3_1_1e4.png" width="192"></td>
<td><img src="examples/layers/conv3_1_1e3.png" width="192"></td>
<td><img src="examples/layers/conv3_1_1e2.png" width="192"></td>
</tr>
<tr>
<td>conv4_1</td>
<td><img src="examples/layers/conv4_1_1e5.png" width="192"></td>
<td><img src="examples/layers/conv4_1_1e4.png" width="192"></td>
<td><img src="examples/layers/conv4_1_1e3.png" width="192"></td>
<td><img src="examples/layers/conv4_1_1e2.png" width="192"></td>
</tr>
<tr>
<td>conv5_1</td>
<td><img src="examples/layers/conv5_1_1e5.png" width="192"></td>
<td><img src="examples/layers/conv5_1_1e4.png" width="192"></td>
<td><img src="examples/layers/conv5_1_1e3.png" width="192"></td>
<td><img src="examples/layers/conv5_1_1e2.png" width="192"></td>
</tr>
</table>


## Setup
#### Dependencies:
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [opencv](http://opencv.org/downloads.html)

#### Optional (but recommended) dependencies:
* [CUDA](https://developer.nvidia.com/cuda-downloads) 7.5+
* [cuDNN](https://developer.nvidia.com/cudnn) 5.0+

#### After installing the dependencies: 
* Download the [VGG-19 model weights](http://www.vlfeat.org/matconvnet/pretrained/) (see the "VGG-VD models from the *Very Deep Convolutional Networks for Large-Scale Visual Recognition* project" section). More info about the VGG-19 network can be found [here](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).
* After downloading, copy the weights file `imagenet-vgg-verydeep-19.mat` to the project directory.

## Usage
### Basic Usage

#### Single Image
1. Copy 1 content image to the default image content directory `./image_input`
2. Copy 1 or more style images to the default style directory `./styles`
3. Run the command:
```
bash stylize_image.sh <path_to_content_image> <path_to_style_image>
```
*Example*:
```
bash stylize_image.sh ./image_input/lion.jpg ./styles/kandinsky.jpg
```
*Note*: Supported image formats include: `.png`, `.jpg`, `.ppm`, `.pgm`

*Note*: Paths to images should not contain the `~` character to represent your home directory; you should instead use a relative path or the absolute path.

#### Single Image or Video Frames
1. Copy content images to the default image content directory `./image_input` or copy video frames to the default video content directory `./video_input`  
2. Copy 1 or more style images to the default style directory `./styles`  
3. Run the command with specific arguments:
```
python neural_style.py <arguments>
```
*Example (Single Image)*:
```
python neural_style.py --content_img golden_gate.jpg \
                       --style_imgs starry-night.jpg \
                       --max_size 1000 \
                       --max_iterations 100 \
                       --original_colors \
                       --device /cpu:0 \
                       --verbose;
```

#### Arguments
* `--content_img`: Filename of the content image. *Example*: `lion.jpg`
* `--content_img_dir`: Relative or absolute directory path to the content image. *Default*: `./image_input`
* `--style_imgs`: Filenames of the style images. To use multiple style images, pass a *space-separated* list.  *Example*: `--style_imgs starry-night.jpg`
* `--style_imgs_weights`: The blending weights for each style image.  *Default*: `1.0` (assumes only 1 style image)
* `--style_imgs_dir`: Relative or absolute directory path to the style images. *Default*: `./styles`
* `--init_img_type`: Image used to initialize the network. *Choices*: `content`, `random`, `style`. *Default*: `content`
* `--max_size`: Maximum width or height of the input images. *Default*: `512`
* `--content_weight`: Weight for the content loss function. *Default*: `5e0`
* `--style_weight`: Weight for the style loss function. *Default*: `1e4`
* `--tv_weight`: Weight for the total variational loss function. *Default*: `1e-3`
* `--temporal_weight`: Weight for the temporal loss function. *Default*: `2e2`
* `--content_layers`: *Space-separated* VGG-19 layer names used for the content image. *Default*: `conv4_2`
* `--style_layers`: *Space-separated* VGG-19 layer names used for the style image. *Default*: `relu1_1 relu2_1 relu3_1 relu4_1 relu5_1`
* `--content_layer_weights`: *Space-separated* weights of each content layer to the content loss. *Default*: `1.0`
* `--style_layer_weights`: *Space-separated* weights of each style layer to loss. *Default*: `0.2 0.2 0.2 0.2 0.2`
* `--original_colors`: Boolean flag indicating if the style is transferred but not the colors.
* `--color_convert_type`: Color spaces (YUV, YCrCb, CIE L\*u\*v\*, CIE L\*a\*b\*) for luminance-matching conversion to original colors. *Choices*: `yuv`, `ycrcb`, `luv`, `lab`. *Default*: `yuv`
* `--style_mask`: Boolean flag indicating if style is transferred to masked regions.
* `--style_mask_imgs`: Filenames of the style mask images (example: `face_mask.png`). To use multiple style mask images, pass a *space-separated* list.  *Example*: `--style_mask_imgs face_mask.png face_mask_inv.png`
* `--noise_ratio`: Interpolation value between the content image and noise image if network is initialized with `random`. *Default*: `1.0`
* `--seed`: Seed for the random number generator. *Default*: `0`
* `--model_weights`: Weights and biases of the VGG-19 network.  Download [here](http://www.vlfeat.org/matconvnet/pretrained/). *Default*:`imagenet-vgg-verydeep-19.mat`
* `--pooling_type`: Type of pooling in convolutional neural network. *Choices*: `avg`, `max`. *Default*: `avg`
* `--device`: GPU or CPU device.  GPU mode highly recommended but requires NVIDIA CUDA. *Choices*: `/gpu:0` `/cpu:0`. *Default*: `/gpu:0`
* `--img_output_dir`: Directory to write output to.  *Default*: `./image_output`
* `--img_name`: Filename of the output image. *Default*: `result`
* `--verbose`: Boolean flag indicating if statements should be printed to the console.

#### Optimization Arguments
* `--optimizer`: Loss minimization optimizer.  L-BFGS gives better results.  Adam uses less memory. *Choices*: `lbfgs`, `adam`. *Default*: `lbfgs`
* `--learning_rate`: Learning-rate parameter for the Adam optimizer. *Default*: `1e0`  

<p align="center">
<img src="examples/equations/plot.png" width="360px">
</p>

* `--max_iterations`: Max number of iterations for the Adam or L-BFGS optimizer. *Default*: `1000`
* `--print_iterations`: Number of iterations between optimizer print statements. *Default*: `50`
* `--content_loss_function`: Different constants K in the content loss function. *Choices*: `1`, `2`, `3`. *Default*: `1` 

<p align="center">
<img src="examples/equations/content.png" width="321px">
</p>


## Acknowledgements

The implementation is based on the projects: 
* Torch (Lua) implementation 'neural-style' by [jcjohnson](https://github.com/jcjohnson)
* Torch (Lua) implementation 'artistic-videos' by [manuelruder](https://github.com/manuelruder)
