import numpy as np
import pickle
import matplotlib.pyplot as plt

import ocr

label_file = './data/train_labels.csv'
labels = np.loadtxt(label_file, dtype=np.object, delimiter=',', skiprows=1)

m, n = labels.shape
errors_name = 0
errors_score = 0
errors_lvl = 0
errors_red = 0
errors_gold = 0
for i in range(m):
	img_name = labels[i, 0]
	img_array = plt.imread(img_name)[...,:3]
	output_dicts = ocr.process_image(img_array, 2)
	for j in range(10):
		print(output_dicts[j]['name'])
		# errors_name += output_dict["name"] != labels[i, j*5]
	import pdb; pdb.set_trace()
	print()