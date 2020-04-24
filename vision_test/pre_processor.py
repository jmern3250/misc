import numpy as np
import cv2
from skimage import transform as tf

import matplotlib.pyplot as plt

SCALE = 5

SCORE_COLOR = np.array([255, 246, 157]).reshape([1,1,3]) #R, G, B
LVL_COLOR = np.array([216, 212, 244]).reshape([1,1,3]) #R, G, B
NAME_COLOR = np.array([253, 253, 253]).reshape([1,1,3]) #R, G, B
RED_STAR_COLOR = np.array([206, 0, 0]).reshape([1,1,3]) #R, G, B
GOLD_STAR_COLOR = np.array([249, 185, 51]).reshape([1,1,3]) #R, G, B
GRAY_STAR_COLOR = np.array([118, 118, 118]).reshape([1,1,3]) #R, G, B

W_SCORE = 150
H_SCORE = 50

W_LVL = 100
H_LVL = 50

W_NAME = 200
H_NAME = 50

W_STAR = 150
H_STAR = 50

SHEAR = 0.25

def pre_process(img_array):
	h, w, _ = img_array.shape

	size = (int(w*SCALE), int(h*SCALE))
	img_array = cv2.resize(img_array.astype(np.uint8), size, interpolation=cv2.INTER_LANCZOS4)
	
	score_map = extract_layer(img_array, SCORE_COLOR, 30, W_SCORE, H_SCORE, SHEAR)
	name_map = extract_layer(img_array, NAME_COLOR, 50, W_NAME, H_NAME, SHEAR)
	lvl_map = extract_layer(img_array, LVL_COLOR, 30, W_LVL, H_LVL, SHEAR)
	red_star_map = extract_layer(img_array, RED_STAR_COLOR, 30, W_STAR, H_STAR, SHEAR, False)
	gold_star_map = extract_layer(img_array, GOLD_STAR_COLOR, 30, W_STAR, H_STAR, SHEAR, False)
	return score_map, name_map, lvl_map, red_star_map, gold_star_map

def extract_layer(img_array, color, d_threshold, w, h, shear, text=True):
	# 1) Binarize Image
	d_score = np.sum(np.abs(img_array.astype(np.float32) - color), axis=-1)
	score_map = d_score <= d_threshold
	# 2) K-means Extraction 
	score_map = extract_window(score_map, w*SCALE, h*SCALE)
	# print(score_map.dtype)
	# plt.imshow(score_map, cmap='gray')
	# plt.show()
	score_map = score_map.astype(np.float32)

	if text:
		# 3) Un-shear (remove italic slant)
		affine_tf = tf.AffineTransform(shear=shear)
		score_map = tf.warp(score_map, inverse_map=affine_tf)
		score_map = np.ceil(score_map).astype(np.uint8)

		# 4) Define Kernel and Close
		kernel_dim = 5
		kernel = np.ones((kernel_dim, kernel_dim),np.uint8)

		score_map = cv2.morphologyEx(score_map, cv2.MORPH_CLOSE, kernel, iterations=2)

		# 5) Erosion
		kernel_dim = 3
		kernel = np.ones((kernel_dim, kernel_dim),np.uint8)
		score_map = cv2.erode(score_map, kernel, iterations=2)

	# 6) Invert Binarization 
	score_map = (1 - score_map)*255
	return score_map

def extract_window(img, w, h): #FIX OVERFLOW
	m, n = img.shape
	x_grid, y_grid = np.meshgrid(np.arange(n), np.arange(m))
	x_weighted = img*x_grid.astype(np.float32)
	y_weighted = img*y_grid.astype(np.float32)
	den = np.sum(img)
	if den == 0:
		return np.zeros((h,w), dtype=np.bool)
	x0 = int(np.sum(x_weighted)/np.sum(img)) - w//2
	y0 = int(np.sum(y_weighted)/np.sum(img)) - h//2
	x1 = x0 + w
	y1 = y0 + h

	mask = np.zeros((m, n))
	mask[y0:y1, x0:x1, ...] = 1

	x_weighted *= mask
	y_weighted *= mask

	den = np.sum(img*mask)
	if den == 0:
		return np.zeros((h,w), dtype=np.bool)
	x0 = int(np.sum(x_weighted)/den) - w//2
	y0 = int(np.sum(y_weighted)/den) - h//2
	x1 = x0 + w
	y1 = y0 + h
	window = img[y0:y1, x0:x1]
	return window