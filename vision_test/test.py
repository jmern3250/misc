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

START = 5
for k in range(m):
	i = START + k 
	print("Starting Image %i" % i)
	img_name = labels[i, 0]
	img_array = plt.imread(img_name)[...,:3]
	if img_name[-3:] == 'png':
		pass
	else:
		try:
			output_dicts = ocr.process_image(img_array, 2)
			for j in range(10):
				print(output_dicts[j]['name'])
				print(output_dicts[j]['score'])
				print(output_dicts[j]['lvl'])
			plt.imshow(img_array)
			plt.show()
		except:
			print("Errored Out")
	print("Image %i Complete" % i)

	if i >= 10:
		break