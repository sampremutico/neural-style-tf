import argparse
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt

def parse_args():
	desc = "Generate masks for stylized objects"  
	parser = argparse.ArgumentParser(description=desc)
	parser.add_argument('--img', type=str,
	    help='Filename of the image (example: lion.jpg)')
	parser.add_argument('--mask_dir', type=str,
		default='data/masks',
		help='Path to mask directory')
	parser.add_argument('--max_size',type=int,
		default=255, help='default size to resize image too')
	args = parser.parse_args()
	return args

def save_mask(mask,mask_filename):
  mask_path = os.path.join(args.mask_dir, mask_filename)
  print(mask_path)
  cv.imwrite(mask_path, mask)

def prep_files():
	img_file = args.img
	dot_idx = img_file.rfind('.')
	slash_idx = img_file.rfind('/')
	mask_file = img_file[slash_idx+1:dot_idx]+"_mask.png"
	return img_file, mask_file

def resize(img):
	h, w, d = img.shape
  	mx = args.max_size
  	# resize if > max size
	if h > w and h > mx:
		w = (float(mx) / float(h)) * w
		img = cv.resize(img, dsize=(int(w), mx), interpolation=cv.INTER_AREA)
	if w > mx:
		h = (float(mx) / float(w)) * h
		img = cv.resize(img, dsize=(mx, int(h)), interpolation=cv.INTER_AREA)
	return img

def segment(img_file):
	img = cv.imread(args.img)
	img = resize(img)
	mask_path = os.path.join("data/masks", "temp.jpg")
	cv.imwrite(mask_path, img)
	print(mask_path)
	exit(0)
	plt.imshow(img)
	plt.show()
	mask = np.zeros(img.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	#rect = (15,5,220,130)
	rect = (1,5, 200,250)
	print(img.shape)
	cv.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
	mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	mask_threshold = np.where((mask==2)|(mask==0),0,255).astype('uint8')
	img = img*mask2[:,:,np.newaxis]
	return img,mask_threshold

def display_mask(mask):
	plt.imshow(mask)
	plt.show()


def main():
	global args
	args = parse_args()
	img_file, mask_file = prep_files()
	masked_img, mask = segment(img_file)
	display_mask(masked_img)
	save_mask(mask,mask_file)

if __name__ == '__main__':
  main()
