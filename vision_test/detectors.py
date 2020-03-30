import numpy as np
import scipy.ndimage
import scipy.misc
import skimage.measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

########## Hyperparameters ##########

SCORE_COLOR = np.array([255, 246, 157]).reshape([1,1,3]) #R, G, B
LVL_COLOR = np.array([216, 212, 244]).reshape([1,1,3]) #R, G, B
NAME_COLOR = np.array([253, 253, 253]).reshape([1,1,3]) #R, G, B
RED_STAR_COLOR = np.array([206, 0, 0]).reshape([1,1,3]) #R, G, B
GOLD_STAR_COLOR = np.array([249, 185, 51]).reshape([1,1,3]) #R, G, B
GRAY_STAR_COLOR = np.array([118, 118, 118]).reshape([1,1,3]) #R, G, B

##### Name #####
name_d_y = 15
name_d_x = 110
name_x_initial = [130, 335, 545, 750, 960]
name_y_initial = [230, 430]
name_box = (175, 15)

##### Score #####
score_d_y = 20
score_d_x = 110
score_x_initial = [130, 335, 545, 750, 960]
score_y_initial = [200, 400]
score_box = (125, 30)

##### Lvl #####
lvl_d_y = 20
lvl_d_x = 110
lvl_x_initial = [130, 335, 545, 750, 960]
lvl_y_initial = [180, 380]
lvl_box = (100, 20)

def extract_layers(pic):
	score_map = np.abs(pic - SCORE_COLOR)
	score_map = np.sum(score_map, axis=-1)  <= 10

	lvl_map = np.abs(pic - LVL_COLOR)
	lvl_map = np.sum(lvl_map, axis=-1)  <= 15

	name_map = np.abs(pic - NAME_COLOR)
	name_map = np.sum(name_map, axis=-1)  <= 10

	star_map0 = np.abs(pic - RED_STAR_COLOR)
	star_map0 = np.sum(star_map0, axis=-1)  <= 15

	star_map1 = np.abs(pic - GOLD_STAR_COLOR)
	star_map1 = np.sum(star_map1, axis=-1)  <= 40

	star_map2 = np.abs(pic - GRAY_STAR_COLOR)
	star_map2 = np.sum(star_map2, axis=-1)  <= 15

	star_map = star_map0 + star_map1 + star_map2
	star_map = star_map >= 1
	
	return score_map, lvl_map, name_map, star_map

def detect_layer(layer, x_initial, y_initial, d_x, d_y):
	m, n = layer.shape
	n_items = len(x_initial)*len(y_initial)
	mask = np.zeros((m, n))
	centroids = []
	for x_c in x_initial:
		for y_c in y_initial:
			mask[y_c-d_y:y_c+d_y, x_c-d_x:x_c+d_x] = 1
			centroids.append((x_c, y_c))
	layer = np.round(layer*mask)
	x_grid, y_grid = np.meshgrid(np.arange(n), np.arange(m))
	for i in range(3):
		##### Assign Pixel Clusters #####
		d = np.zeros((m,n,n_items))
		for j, centroid in enumerate(centroids):
			d_x_s = (x_grid - centroid[0])**2
			d_y_s = (y_grid - centroid[1])**2
			d_c = np.sqrt(d_x_s + d_y_s)
			d[...,j] = d_c
		idxs = np.argmin(d, axis=-1) + 1.0 # 1 index the centroids
		idxs *= layer # Mask out the zero-pixels
		centroids_old = centroids
		centroids = []
		##### Update Centroid Location #####
		for j, centroid in enumerate(centroids_old):
			grid_idxs = np.argwhere(idxs == (j+1))
			centroid_rc = np.mean(grid_idxs, axis=0)
			centroids.append((centroid_rc[1], centroid_rc[0]))
	return np.array(centroids).astype(np.int32)
			
def display_w_boxes(pic, centroids, box_size):
	w, h = box_size
	fig, ax = plt.subplots(1)
	ax.imshow(pic, cmap='gray')
	for centroid in centroids:
		x, y = centroid
		x -= w//2
		y -= h//2
		rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
	plt.show()


pic_file = './data/pic1.jpg'
pic = plt.imread(pic_file)
# import pdb; pdb.set_trace()
pic_r = resize(pic, (540,1140)) #, preserve_range=True)

score_map, lvl_map, name_map, star_map = extract_layers(pic)

# plt.figure()
# plt.imshow(pic_r)
# plt.figure()
# plt.imshow(score_map, cmap='gray')
# plt.figure()
# plt.imshow(lvl_map, cmap='gray')
# plt.figure()
# plt.imshow(name_map, cmap='gray')
# plt.figure()
# plt.imshow(star_map, cmap='gray')
# plt.show()
name_map_r = resize(name_map.astype(np.float32), (540,1140)) #, preserve_range=True)
# plt.imshow(name_map.astype(np.float32), cmap='gray')
# plt.show()
# plt.imshow(name_map_r, cmap='gray')
# plt.show()
# import pdb; pdb.set_trace()
centroids = detect_layer(name_map_r, name_x_initial, name_y_initial, name_d_x, name_d_y)
# import pdb; pdb.set_trace()
display_w_boxes(pic_r, centroids, name_box)