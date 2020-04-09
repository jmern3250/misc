import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pytesseract
from skimage import transform as tf

SCORE_COLOR = np.array([255, 246, 157]).reshape([1,1,3]) #R, G, B
LVL_COLOR = np.array([216, 212, 244]).reshape([1,1,3]) #R, G, B
NAME_COLOR = np.array([253, 253, 253]).reshape([1,1,3]) #R, G, B

W_SCORE = 150
H_SCORE = 50

W_LVL = 100
H_LVL = 50

W_NAME = 150
H_NAME = 50

CORR_DICT = {'S':'5',
			 'G':'6'}

img_array = plt.imread('./data/img6.png')[...,:3]
img_array *= 255

def extract_window(img, w, h):
	m, n = img.shape
	x_grid, y_grid = np.meshgrid(np.arange(n), np.arange(m))
	x_weighted = img*x_grid.astype(np.float32)
	y_weighted = img*y_grid.astype(np.float32)
	x0 = int(np.sum(x_weighted)/np.sum(img)) - w//2
	y0 = int(np.sum(y_weighted)/np.sum(img)) - h//2
	x1 = x0 + w
	y1 = y0 + h

	mask = np.zeros((m, n))
	mask[y0:y1, x0:x1, ...] = 1

	x_weighted *= mask
	y_weighted *= mask
	x0 = int(np.sum(x_weighted)/np.sum(img*mask)) - w//2
	y0 = int(np.sum(y_weighted)/np.sum(img*mask)) - h//2
	x1 = x0 + w
	y1 = y0 + h
	window = img[y0:y1, x0:x1, ...]
	return window

def correct(in_string):
	blacklist = set(CORR_DICT.keys())
	out_string = ''
	for char in in_string: 
		if char in blacklist:
			out_string += CORR_DICT[char]
		else:
			out_string += char
	return out_string

# ##### Score #####
# kernel_dim = 5
# kernel = np.ones((kernel_dim, kernel_dim),np.float32)/kernel_dim**2

# d_score = np.sum(np.abs(img_array - SCORE_COLOR), axis=-1)
# score_map = d_score <= 30
# score_map = extract_window(score_map, W_SCORE, H_SCORE)
# score_map = (1 - score_map).astype(np.uint8)*255
# score_map = cv2.filter2D(score_map,-1,kernel)
# plt.imshow(score_map, cmap='gray')
# plt.show()

# score_string = pytesseract.image_to_string(score_map, lang='eng', config='--oem 1 --psm 6')
# print("Score: ", score_string)

##### Lvl #####
# lvl_map = (1 - lvl_map).astype(np.uint8)*255
# lvl_map = cv2.filter2D(lvl_map,-1,kernel)#/255
# lvl_map = np.floor(lvl_map)*255
# lvl_map = np.stack([lvl_map]*3, axis=-1).astype(np.uint8)
h, w, _ = img_array.shape
SCALE = 5
size = (int(w*SCALE), int(h*SCALE))
img_array = cv2.resize(img_array, size, interpolation=cv2.INTER_LANCZOS4)

# 1) Binarize Image
d_lvl = np.sum(np.abs(img_array - LVL_COLOR), axis=-1)
lvl_map = d_lvl <= 30

# 2) K-means Extraction 
lvl_map = extract_window(lvl_map, W_LVL*SCALE, H_LVL*SCALE)
lvl_map = lvl_map.astype(np.float32)

# 3) Un-shear (remove italic slant)
affine_tf = tf.AffineTransform(shear=0.25)
lvl_map = tf.warp(lvl_map, inverse_map=affine_tf)
lvl_map = np.ceil(lvl_map).astype(np.uint8)

# 4) Define Kernel and Close
kernel_dim = 3
kernel = np.ones((kernel_dim, kernel_dim),np.uint8)

lvl_map = cv2.morphologyEx(lvl_map, cv2.MORPH_CLOSE, kernel)
lvl_map = cv2.morphologyEx(lvl_map, cv2.MORPH_CLOSE, kernel)

# 5) Erosion
lvl_map = cv2.erode(lvl_map, kernel, iterations=1)
# lvl_map = cv2.morphologyEx(lvl_map, cv2.MORPH_OPEN, kernel)

# 6) Invert Binarization 
lvl_map = (1 - lvl_map)*255
plt.figure()
plt.imshow(lvl_map, cmap='gray')
plt.show()

# 7) Tesseract OCR
lvl_string = pytesseract.image_to_string(lvl_map, lang='eng', config='--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789SG')

# 8) Swap incorrect characters
lvl_string = correct(lvl_string)
print("lvl: ", lvl_string)


# ##### Name #####
# h, w, _ = img_array.shape
# SCALE = 3.0
# size = (int(w*SCALE), int(h*SCALE))
# img_array = cv2.resize(img_array, size, interpolation=cv2.INTER_CUBIC)

# kernel_dim = 3
# kernel = np.ones((kernel_dim, kernel_dim),np.float32)/kernel_dim**2
# # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

# d_lvl = np.sum(np.abs(img_array - LVL_COLOR), axis=-1)
# lvl_map_1 = d_lvl <= 60
# lvl_map = d_lvl <= 30
# lvl_map = (1 - lvl_map).astype(np.uint8)*255
# # lvl_map = cv2.filter2D(lvl_map,-1,kernel)

# d = pytesseract.image_to_data(lvl_map, lang='eng', config='--oem 1 --psm 6', output_type=pytesseract.Output.DICT)

# d_l = 0
# i = -1
# x, y, w, h = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])

# lvl_map_1 = lvl_map_1[y-d_l:y+h+d_l, x-d_l:x+w+d_l]
# lvl_map_1 = (1 - lvl_map_1).astype(np.uint8)*255
# # lvl_map_1 = cv2.filter2D(lvl_map_1,-1,kernel)

# # import pdb; pdb.set_trace()
# lvl_map_1 = embed_box(lvl_map_1, 0.5)
# # fig, ax = plt.subplots(1)
# # ax.imshow(lvl_map_1, cmap='gray')
# # plt.show()

# lvl_string = pytesseract.image_to_string(img_array.astype(np.uint8), lang='eng', config='--oem 3 --psm 3 -c tessedit_char_whitelist=LVL0123456789 load_system_dawg=false load_freq_dawg=false')
# print(d)
# print("Level: ", lvl_string)

# ##### Name #####
# kernel_dim = 3
# kernel = np.ones((kernel_dim, kernel_dim),np.float32)/kernel_dim**2
# d_name = np.sum(np.abs(img_array - NAME_COLOR), axis=-1)
# name_map = d_name <= 50
# name_map = extract_window(name_map, W_NAME, H_NAME)
# name_map = (1 - name_map).astype(np.uint8)*255
# name_map = cv2.filter2D(name_map,-1,kernel)
# plt.imshow(name_map, cmap='gray')
# plt.show()

# name_string = pytesseract.image_to_string(name_map, lang='eng', config='--oem 1 --psm 6')
# print("Name: ", name_string)
