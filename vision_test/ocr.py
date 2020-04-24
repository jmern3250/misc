import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract

import detector
import pre_processor
import spelling

CORR_DICT = {'S':'5',
			 'G':'6',
			 'U':'0'}

def process_image(img_array, n_rows=None):
	img_windows = detector.detect(img_array, n_rows)
	output_dicts = []
	for window in img_windows:
		score_map, name_map, lvl_map, red_star_map, gold_star_map = pre_processor.pre_process(window)
		score = pytesseract.image_to_string(score_map, lang='eng', config='--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789SG')
		score = num_correct(score)
		lvl = pytesseract.image_to_string(lvl_map, lang='eng', config='--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789SGULV')
		lvl = lvl.replace("L", "")
		lvl = lvl.replace("V", "")
		lvl = num_correct(lvl)
		name = pytesseract.image_to_string(name_map, lang='eng', config='--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ')
		name = spelling.correction(name)
		params = cv2.SimpleBlobDetector_Params()
		params.filterByArea = True
		params.minArea = 500
		params.filterByCircularity = False
		params.filterByConvexity = False
		params.filterByInertia = False
		blob_detector = cv2.SimpleBlobDetector_create(params)
		red_stars = detect_stars(red_star_map, blob_detector)
		gold_stars = detect_stars(gold_star_map, blob_detector)
		output_dict = {'score':score,
					   'lvl':lvl,
					   'name':name,
					   'red_stars':red_stars,
					   'gold_stars':gold_stars
						}
		output_dicts.append(output_dict)
	return output_dicts

def detect_stars(img, blob_detector):
	keypoints = blob_detector.detect(img.astype(np.uint8))
	return len(keypoints)

def num_correct(in_string):
	blacklist = set(CORR_DICT.keys())
	out_string = ''
	for char in in_string: 
		if char in blacklist:
			out_string += CORR_DICT[char]
		else:
			out_string += char
	return out_string

if __name__ == "__main__":
	img_array = plt.imread('./data/archive/pic2.jpg')[...,:3]
	output_dict = process_image(img_array, 2)