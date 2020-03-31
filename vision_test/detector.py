import numpy as np
import matplotlib.pyplot as plt
import cv2

W = 1250
H_MAX = 650
ALPHA = 4.951
BETA = -0.423

B_H = 100
B_W = 175

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

	# full_map = score_map + lvl_map + name_map + star_map
	# full_map = full_map.astype(np.float32)*255.
	full_map = cv2.Canny(img, 100, 200).astype(np.float32)
	# import pdb; pdb.set_trace()
	full_map = full_map[:H_MAX, ...]

	if n_rows is None:
		ar = h/w
		n_rows = np.round(ALPHA*ar + BETA).astype(np.int32)
	elif isnumeric(n_rows):
		n_rows = int(n_rows)
	else:
		raise ValueError('n_rows must be a numeric value or "None"')

	c_y, m, b = detect_rows(full_map, n_rows, 10)
	c_x = detect_cols(full_map)

	return segment_image(img, c_x, c_y)

def detect_rows(layer, n_rows, n_its):
	layer = (layer==255).astype(np.float64)
	m, n = layer.shape
	cur_slope = m/2
	cur_intercept = m/2/2 + 20 
	_, y_grid = np.meshgrid(np.arange(n), np.arange(m))
	y_grid = np.expand_dims(y_grid, axis=-1)
	d_intercept = 0 
	d_slope = 0 
	for i in range(n_its):
		intercepts = 2*(np.arange(7) - 3.) + np.clip(d_intercept, -5, 5) + cur_intercept 
		slopes = 2*(np.arange(7) - 3.) + np.clip(d_slope, -5, 5) + cur_slope 
		min_error = np.inf
		d_slope = 0
		d_intercept = 0 
		for intercept in intercepts:
			for slope in slopes:
				centroids = np.arange(2)*slope + intercept 
				centroids = centroids.reshape((1, 1, 2))
				##### Assign Pixel Clusters #####
				d = np.abs(y_grid - centroids)
				d_min = np.min(d, axis=-1)
				d_min *= layer
				mask = d_min >= 20
				d_min *= mask.astype(np.float32)
				error = np.sum(d_min**2)
				if error < min_error:
					d_slope += slope - cur_slope
					d_intercept += intercept - cur_intercept
					min_error = error
					cur_slope = slope
					cur_intercept = intercept
		if d_slope == 0 and d_intercept == 0: 
			break

	c_rows = np.arange(n_rows)*cur_slope + cur_intercept
	return c_rows, cur_slope, cur_intercept

def detect_cols(layer):
	layer = (layer==255).astype(np.float64)
	m, n = layer.shape
	cur_intercept = n/5/2
	cur_slope = n/5
	x_grid, _ = np.meshgrid(np.arange(n), np.arange(m))
	x_grid = np.expand_dims(x_grid, axis=-1)
	d_intercept = 0 
	d_slope = 0 
	for i in range(10):
		intercepts = 2*(np.arange(7) - 3.) + cur_intercept + np.clip(d_intercept, -5, 5)
		slopes = 2*(np.arange(7) - 3.) + cur_slope + np.clip(d_slope, -5, 5)
		min_error = np.inf
		d_slope = 0
		d_intercept = 0 
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
					d_slope += slope - cur_slope
					d_intercept += intercept - cur_intercept
					min_error = error
					cur_slope = slope
					cur_intercept = intercept
		if d_slope == 0 and d_intercept == 0: 
			break
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

def segment_image(img, c_x, c_y):
	imgs = []
	for y in c_y:
		for x in c_x:
			start_y = int(y-B_H//2)
			end_y = start_y + B_H
			start_x = int(x-B_W//2)
			end_x = start_x + B_W
			img_seg = img[start_y:end_y, start_x:end_x, ...]
			plt.imshow(img_seg)
			plt.show()
			imgs.append(img_seg)
	return imgs

# def segment_image(img, n_rows):
# 	imgs = []
# 	m, n, _ = img.shape
# 	ys = np.linspace(0, m, n_rows+1)
# 	xs = np.linspace(0, n, 6)
# 	for i in range(n_rows):
# 		for j in range(5):
# 			x_start = int(xs[j])
# 			x_end = int(xs[j + 1])
# 			y_start = int(ys[i])
# 			y_end = int(ys[i + 1])
# 			img_seg = img[y_start:y_end, x_start:x_end, ...]
# 			plt.imshow(img_seg)
# 			plt.show()
# 			imgs.append(img_seg)
# 	return imgs

if __name__ == "__main__":
	# img = cv2.imread('data/pic3.jpg')
	img = cv2.imread('data/pic5.png')
	detect(img)
	
