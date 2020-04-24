import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import copy

import detector

AR = 0.474
AR_THRESH = 0.08
W_MIN = 1000

BLACKLIST = "-.() "

def parse_frames(label_file, output_file):
	labels = np.loadtxt(label_file, dtype=np.object, delimiter=',', skiprows=1)
	data_list = []
	m, n = labels.shape
	names = np.unique(labels[:,1])
	name_idx = {}
	for i, name in enumerate(names):
		name_idx[i] = name
	for i in range(m):
		print("Frame %i of %i" % (i+1, m))
		img = labels[i, 0]
		add_img = True
		try: 
			# img_array = plt.imread(img)
			img_array = cv2.imread(img)
			img_array = np.flip(img_array[...,:3], axis=-1)
			h, w, _ = img_array.shape
			ar = float(h)/float(w)
			if w <=  W_MIN:
				add_img = False
			if np.abs(ar - AR) >= AR_THRESH:
				add_img = False
		except:
			add_img = False
		if add_img:
			char_imgs = detector.detect(img_array)
			label_row = labels[i,1:]
			char_names = label_row[::5].tolist()
			char_lvls = label_row[1::5].astype(np.float32).tolist()
			char_pwrs = label_row[2::5].astype(np.float32).tolist()
			char_gold_stars = label_row[3::5].astype(np.float32).tolist()
			char_red_stars = label_row[4::5].astype(np.float32).tolist()

			data_dict = {'imgs':char_imgs,
						 'names':char_names,
						 'lvls':char_lvls,
						 'pwrs':char_pwrs,
						 'gold':char_gold_stars,
						 'red':char_red_stars}
			print("Adding Frame")
			data_list.append(data_dict)
	with open(output_file, 'wb') as f:
		pickle.dump((data_list, name_idx), f)

def parse_names(name_file, output_file):
	org_names = np.loadtxt(name_file, dtype=np.object, delimiter=',', skiprows=1).tolist()
	alph_names = []
	for name in org_names:
		new_name = copy.deepcopy(name)
		for char in BLACKLIST:
			new_name = new_name.replace(char, "")
		alph_names.append(new_name)
	with open(output_file, 'wb') as f:
		pickle.dump((org_names, alph_names), f)

if __name__ == "__main__":
	# label_file = './data/train_labels.csv'
	output_file = './data/train_data.p'
	# # parse_frames(label_file, './data/train_data.p')
	with open(output_file, 'rb') as f:
		parsed_data = pickle.load(f)
	######################
	fwd_dict = parsed_data[1]
	keys = fwd_dict.keys()
	vals = fwd_dict.values()
	inv_dict = dict(zip(vals, keys))

	# parsed_data = (parsed_data[0], fwd_dict, inv_dict)
	# with open(output_file, 'wb') as f:
	# 	pickle.dump(parsed_data, f)
	# with open('./name_dict.p', 'wb') as f:
	# 	pickle.dump(fwd_dict, f)
	# import pdb; pdb.set_trace()
	######################
	# parsed_data = parsed_data[0]
	# plt.figure()
	# for item in parsed_data:
	# 	images = item['imgs']
	# 	names = item['names']
	# 	for i, img in enumerate(images):
	# 		# filename = './data/img' + str(i) + '.png'
	# 		# plt.imsave(filename, img)
	# 		print(names[i])
	# 		plt.imshow(img)
	# 		plt.pause(0.5)

	# # parse_names('./data/names.csv', './data/names.p')