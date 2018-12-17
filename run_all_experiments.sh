for style_layer_group in 1 2 3
	do
	for iters in 1000
		do
		    echo "Using style layer group $style_layer_group"
		    python neural_style.py --content_img data/tiny_raw/cb2/cb2_with_background.jpeg --style_imgs data/tiny_raw/eames/eames_chair_solid_background.jpeg --style_imgs_dir . --content_img_dir . --max_size 255  --max_iterations $iters --print_iterations 1 --device /cpu:0 --verbose --style_mask --style_mask_imgs cb2_with_background_mask.png --content_mask --content_mask_img eames_chair_solid_background_mask.png --img_name dual_mask_cb2_eames_$iters_$style_layer_group --style_layer_group $style_layer_group
	    done
	    do
		    echo "Using style layer group $style_layer_group"
		    python neural_style.py --content_img data/tiny_raw/ikea/ikea_chair.jpeg --style_imgs data/tiny_raw/etc/west_elm.jpg --style_imgs_dir . --content_img_dir . --max_size 255  --max_iterations $iters --print_iterations 10 --device /cpu:0 --verbose --style_mask --style_mask_imgs ikea_chair_mask.png --content_mask --content_mask_img west_elm_mask.png --img_name dual_mask_ikea_westelm_$iters_$style_layer_group --style_layer_group $style_layer_group
	    done
	done