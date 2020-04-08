import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract

SCORE_COLOR = np.array([255, 246, 157]).reshape([1,1,3]) #R, G, B
LVL_COLOR = np.array([216, 212, 244]).reshape([1,1,3]) #R, G, B
NAME_COLOR = np.array([253, 253, 253]).reshape([1,1,3]) #R, G, B

img_array = plt.imread('./data/img1.png')[...,:3]
img_array *= 255

kernel_dim = 4
kernel = np.ones((kernel_dim, kernel_dim),np.float32)/kernel_dim**2

def extract_window(img, w, h):
	##### First center based on full image #####
	m, n = img.shape
	x_grid, y_grid = np.meshgrid(np.arange(n), np.arange(m))
	x_weighted = img*x_grid
	y_weighted = img*y_grid
	import pdb; pdb.set_trace()
	print()
##### Score #####
d_score = np.sum(np.abs(img_array - SCORE_COLOR), axis=-1)
score_map = (1 - (d_score <= 30)).astype(np.uint8)*255
score_map = cv2.filter2D(score_map,-1,kernel)
plt.imshow(score_map, cmap='gray')
plt.show()

score_string = pytesseract.image_to_string(score_map, config='--oem 1 -c outputbase=digits')
print("Score: ", score_string)

# ##### Name #####
# d_lvl = np.sum(np.abs(img_array - LVL_COLOR), axis=-1)
# lvl_map = (1 - (d_lvl <= 40)).astype(np.uint8)*255
# lvl_map = cv2.filter2D(lvl_map,-1,kernel)
# plt.imshow(lvl_map, cmap='gray')
# plt.show()

# lvl_string = pytesseract.image_to_string(lvl_map)
# print("Level: ", lvl_string)

# ##### Name #####
# d_name = np.sum(np.abs(img_array - NAME_COLOR), axis=-1)
# name_map = (1 - (d_name <= 60)).astype(np.uint8)*255
# name_map = cv2.filter2D(name_map,-1,kernel)
# plt.imshow(name_map, cmap='gray')
# plt.show()

# name_string = pytesseract.image_to_string(name_map)
# print("Name: ", name_string)
