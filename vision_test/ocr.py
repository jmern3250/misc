import numpy as np
import matplotlib.pyplot as plt
import cv2
import pytesseract

import network
import detector
import pre_processor
import spelling

import argparse
import csv

CORR_DICT = {'S':'5',
			 'G':'6',
			 'U':'0'}

GRAPH = network.OCRNet(137, 217, GPU=True)
GRAPH.build_ocr_net()
GRAPH.restore_model('./models/net_1_15')

HEADINGS = ['id','label','unlocked','power','level','gearLevel','yellowStars','redStars','basic','special','ultimate','passive']

def process_image(img_array, n_rows=None):
	img_windows = detector.detect(img_array, n_rows)
	output_dicts = []
	for window in img_windows:
		score_map, name_map, lvl_map, red_star_map, gold_star_map = pre_processor.pre_process(window)
		score = pytesseract.image_to_string(score_map, lang='eng', config='--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789SG')
		score = num_correct(score)
		name, lvl, red_stars, gold_stars = GRAPH.encode_image(window)
		######## Classical Name Method #########
		# name = pytesseract.image_to_string(name_map, lang='eng', config='--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ')
		# name = spelling.correction(name)
		########## Classical methods ###########
		# lvl = pytesseract.image_to_string(lvl_map, lang='eng', config='--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789SGULV')
		# lvl = lvl.replace("L", "")
		# lvl = lvl.replace("V", "")
		# lvl = num_correct(lvl)
		# params = cv2.SimpleBlobDetector_Params()
		# params.filterByArea = True
		# params.minArea = 500
		# params.filterByCircularity = False
		# params.filterByConvexity = False
		# params.filterByInertia = False
		# blob_detector = cv2.SimpleBlobDetector_create(params)
		# red_stars = detect_stars(red_star_map, blob_detector)
		# gold_stars = detect_stars(gold_star_map, blob_detector)
		########################################
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

def load_image(image_file):
	img_array = plt.imread(image_file)[...,:3]
	max_val = np.amax(img_array)
	if max_val != 255:
		img_array *= 255.0
	else:
		img_array = img_array.astype(np.float32)
	return img_array

def write_csv(result_dicts, output_filename):
	with open(output_filename, 'w', newline='') as csvfile:
	    writer = csv.writer(csvfile, delimiter=',')
	    writer.writerow(HEADINGS)
	    for i, result in enumerate(result_dicts):
	    	row = ['', result['name'], '', result['score'], result['lvl'], '', result['gold_stars'], result['red_stars']]
	    	writer.writerow(row)

def main(args):
	img_array = load_image(args.input_filename)
	result_dicts = process_image(img_array, args.n_rows)
	write_csv(result_dicts, args.output_filename)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Specify Input and Output Files')
	parser.add_argument('input_filename', type=str, help='an integer for the accumulator')
	parser.add_argument('--output_filename', type=str, default='./output.csv')
	parser.add_argument('--n_rows', default=None)
	args = parser.parse_args()
	main(args)