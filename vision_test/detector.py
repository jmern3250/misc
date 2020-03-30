import numpy as np
import matplotlib.pyplot as plt
import cv2

W = 1250
ALPHA = 4.951
BETA = -0.423

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
	img = np.flip(img, axis=-1)

	score_map, lvl_map, name_map, star_map = extract_layers(img)

	if n_rows is None:
		ar = h/w
		n_rows = np.round(ALPHA*ar + BETA).astype(np.int32)
	elif isnumeric(n_rows):
		n_rows = int(n_rows)
	else:
		raise ValueError('n_rows must be a numeric value or "None"')

	h, w, _ = img.shape
	m0 = b0 = h//(n_rows + 1.)
	c_y_score, m, b = detect_rows(score_map.astype(np.float64)*255., r_est_linear, 10, m0, b0)
	c_x = detect_cols(score_map.astype(np.float64)*255.)

	plt.imshow(img)
	for x in c_x:
		for y in c_y:
			plt.plot(x, y, 'ro')
	plt.show()


def detect_rows(layer, n_rows, n_its, m0, b0):
	layer = (layer==255).astype(np.float64)
	m, n = layer.shape
	cur_intercept = b0
	cur_slope = m0
	_, y_grid = np.meshgrid(np.arange(n), np.arange(m))
	y_grid = np.expand_dims(y_grid, axis=-1)
	for i in range(n_its):
		intercepts = [cur_intercept - 5, cur_intercept - 1 , cur_intercept, cur_intercept + 1, cur_intercept + 5]
		slopes = [cur_slope - 5, cur_slope - 1 , cur_slope, cur_slope + 1, cur_slope + 5]
		min_error = np.inf
		for intercept in intercepts:
			for slope in slopes:
				centroids = np.arange(n_rows)*slope + intercept 
				centroids = centroids.reshape((1, 1, n_rows))
				##### Assign Pixel Clusters #####
				d = np.square(y_grid - centroids)
				d_min = np.min(d, axis=-1)
				d_min *= layer
				error = np.sum(d_min**2)
				if error < min_error:
					min_error = error
					cur_slope = slope
					cur_intercept = intercept

	c_rows = np.arange(n_rows)*cur_slope + cur_intercept
	return c_rows, cur_slope, cur_intercept

def detect_cols(layer):
	layer = (layer==255).astype(np.float64)
	m, n = layer.shape
	cur_intercept = n//(5. + 1)
	cur_slope = n//(5 + 1.)
	x_grid, _ = np.meshgrid(np.arange(n), np.arange(m))
	x_grid = np.expand_dims(x_grid, axis=-1)
	for i in range(10):
		intercepts = [cur_intercept - 10, cur_intercept - 1 , cur_intercept, cur_intercept + 1, cur_intercept + 10]
		slopes = [cur_slope - 10, cur_slope - 1 , cur_slope, cur_slope + 1, cur_slope + 10]
		min_error = np.inf
		max_score = 0
		for intercept in intercepts:
			for slope in slopes:
				centroids = np.arange(5)*slope + intercept 
				centroids = centroids.reshape((1, 1, 5))
				##### Assign Pixel Clusters #####
				d = np.square(x_grid - centroids)
				d_min = np.min(d, axis=-1)
				d_min *= layer
				error = np.sum(d_min**2)
				if error < min_error:
					min_error = error
					cur_slope = slope
					cur_intercept = intercept

	c_cols = np.arange(5)*cur_slope + cur_intercept
	return c_cols

def row_scores(layer, c_rows):
	m, n  = layer.shape
	layer = (layer==255).astype(np.float64)
	row_bottom = 0 
	row_scores = []
	n_rows = len(c_rows)
	_, y_grid = np.meshgrid(np.arange(n), np.arange(m))
	for i in range(n_rows):
		row_top = row_bottom
		row_c = c_rows[i]
		if i == (n_rows -1):
			row_bottom = int(m)
		else:
			next_row = c_rows[i+1]
			d_row = (next_row - row_c)//2
			row_bottom = int(row_c + d_row)
		row_h = row_bottom - row_top
		scores = 1/(1 + np.abs(y_grid - row_c))
		scores *= layer
		score = np.sum(scores[row_top:row_bottom, ...])/row_h
		row_scores.append(score)
	return np.amin(row_scores)


def extract_layers(pic):
	score_map = np.abs(pic - SCORE_COLOR)
	score_map = np.sum(score_map, axis=-1)  <= 15

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

def detect_layer(layer, n_rows, d_x, d_y):
	m, n = layer.shape
	y_initial = np.linspace(0, m, n_rows + 2)[1:-1]
	y_initial = y_initial.astype(np.int32)
	x_initial = np.linspace(0, n, 7)[1:-1]
	x_initial = x_initial.astype(np.int32)

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


if __name__ == "__main__":
	img = cv2.imread('data/pic3.jpg')
	# img = cv2.imread('data/pic4.png')
	detect(img)
	
