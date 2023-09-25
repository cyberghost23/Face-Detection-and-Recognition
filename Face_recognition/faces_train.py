# Author: Walid Sharaiyra
# Face Recognition using openCV-python
# Date  : 14 March 2022
# AI Training Program 

import os
import cv2
from PIL import Image
import numpy as np
import pickle

faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "images")
rec = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path))
			# print(label, path)
			if label in label_ids:
				pass
			else:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			print(label_ids)
			pil_image = Image.open(path).convert("L")
			size = (320, 320)
			final_image = pil_image.resize(size, Image.ANTIALIAS)
			image_array = np.array(final_image, "uint8")
			# print(image_array)
			faces = faceCascade.detectMultiScale(
				image_array,
			    scaleFactor=1.06,
			    minNeighbors=5,
			    minSize=(30, 30)
			)
			for (x, y, w, h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_labels.append(id_)


# print(y_labels)
# print(x_train)

with open("labels.pickle", "wb") as f:
	pickle.dump(label_ids, f)

rec.train(x_train, np.array(y_labels))
rec.save("trainer.yml")