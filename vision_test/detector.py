import numpy as np
import matplotlib.pyplot as plt
import cv2

W = 1600
H = 757
ALPHA = 4.951
BETA = -0.423

B_H = int(120*1.6/1.4)
B_W = int(190*1.6/1.4)

SCORE_COLOR = np.array([255, 246, 157]).reshape([1,1,3]) #R, G, B
LVL_COLOR = np.array([216, 212, 244]).reshape([1,1,3]) #R, G, B
NAME_COLOR = np.array([253, 253, 253]).reshape([1,1,3]) #R, G, B
RED_STAR_COLOR = np.array([206, 0, 0]).reshape([1,1,3]) #R, G, B
GOLD_STAR_COLOR = np.array([249, 185, 51]).reshape([1,1,3]) #R, G, B
GRAY_STAR_COLOR = np.array([118, 118, 118]).reshape([1,1,3]) #R, G, B

def detect(img, n_rows=None):
	h, w, _ = img.shape
	scale_factor = W/w
	target_size = (int(w*scale_factor), int(h*scale_factor))

	img = cv2.resize(img, target_size)
	img = img[:760, ..., :3]
	h, w, _ = img.shape

	score_map, lvl_map, name_map, star_map = extract_layers(img)

	full_map = score_map + star_map + lvl_map #+ name_map 
	full_map = full_map.astype(np.float32)*255.

	if n_rows is None:
		ar = h/w
		n_rows = np.round(ALPHA*ar + BETA).astype(np.int32)
	elif isinstance(n_rows, (int, float)):
		n_rows = int(n_rows)
	else:
		raise ValueError('n_rows must be a numeric value or "None"')

	c_y = detect_rows(full_map, n_rows, 5)
	c_x = detect_cols(full_map, 5)
	return segment_image(img, c_x, c_y)

def detect_rows(layer, n_rows, n_its):
	layer = (layer==255).astype(np.int64)
	m, n = layer.shape
	centroids = np.linspace(0, m, n_rows + 2)[1:-1]
	_, y_grid = np.meshgrid(np.arange(n), np.arange(m))
	y_grid = np.expand_dims(y_grid, axis=-1)
	for i in range(n_its):
		centroids = centroids.reshape((1, 1, n_rows))
		##### Assign Pixel Clusters #####
		d = np.square(y_grid - centroids)
		idxs = np.argmin(d, axis=-1) + 1
		idxs *= layer
		centroids_old = centroids.squeeze()
		centroids = []
		##### Update Centroid Location #####
		for j, centroid in enumerate(centroids_old):
			grid_idxs = np.argwhere(idxs == (j+1))
			centroid_c = np.mean(grid_idxs, axis=0)[0]
			centroids.append(int(centroid_c))
		centroids = np.array(centroids)

	return centroids

def detect_cols(layer, n_its):
	layer = (layer==255).astype(np.int64)
	m, n = layer.shape
	centroids = np.linspace(0, n, 5 + 2)[1:-1]
	x_grid, _ = np.meshgrid(np.arange(n), np.arange(m))
	x_grid = np.expand_dims(x_grid, axis=-1)
	for i in range(n_its):
		centroids = centroids.reshape((1, 1, 5))
		##### Assign Pixel Clusters #####
		d = np.square(x_grid - centroids)
		idxs = np.argmin(d, axis=-1) + 1
		idxs *= layer
		centroids_old = centroids.squeeze()
		centroids = []
		##### Update Centroid Location #####
		for j, centroid in enumerate(centroids_old):
			grid_idxs = np.argwhere(idxs == (j+1))
			centroid_c = np.mean(grid_idxs, axis=0)[1]
			centroids.append(int(centroid_c))
		centroids = np.array(centroids)

	return centroids

def extract_layers(pic, d_threshold=0):
	score_map = np.abs(pic - SCORE_COLOR)
	score_map = np.sum(score_map, axis=-1)  <= 15 + d_threshold

	lvl_map = np.abs(pic - LVL_COLOR)
	lvl_map = np.sum(lvl_map, axis=-1)  <= 15 + d_threshold

	name_map = np.abs(pic - NAME_COLOR)
	name_map = np.sum(name_map, axis=-1)  <= 15 + d_threshold

	star_map0 = np.abs(pic - RED_STAR_COLOR)
	star_map0 = np.sum(star_map0, axis=-1)  <= 15 + d_threshold

	star_map1 = np.abs(pic - GOLD_STAR_COLOR)
	star_map1 = np.sum(star_map1, axis=-1)  <= 40 + d_threshold

	star_map2 = np.abs(pic - GRAY_STAR_COLOR)
	star_map2 = np.sum(star_map2, axis=-1)  <= 15 + d_threshold

	star_map = star_map0 + star_map1 + star_map2
	star_map = star_map >= 1
	
	return score_map, lvl_map, name_map, star_map

def segment_image(img, c_x, c_y):
	imgs = []
	for y in c_y:
		for x in c_x:
			start_y = int(y-B_H//2)
			end_y = start_y + B_H
			start_x = int(x-B_W//2)
			end_x = start_x + B_W
			img_seg = img[start_y:end_y, start_x:end_x, ...]
			imgs.append(img_seg)
	return imgs

if __name__ == "__main__":
	img = cv2.imread('data/pic3.jpg')
	# img = cv2.imread('data/pic4.png')
	detect(img)
	
