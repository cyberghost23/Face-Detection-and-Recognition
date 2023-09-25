# Author: Walid Sharaiyra
# Face Recognition using openCV-python
# Date  : 14 March 2022
# AI Face Recognition Program

import cv2
import numpy as np
import pickle

faceCascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("trainer.yml")

labels = {}
with open("labels.pickle", "rb") as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}


cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	faces = faceCascade.detectMultiScale(
	    gray,
	    scaleFactor=1.42,
	    minSize=(30, 30),
	    # minNeighbors=5
	)
	# print("Found {0} faces!".format(len(faces)))

	for (x, y, w, h) in faces:
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = frame[y:y+h, x:x+w]
	    id_, conf = rec.predict(roi_gray)

	    if conf >= 45 and conf <= 85:
	    	text = labels[id_]
	    	# print(labels[id_] + " >> " + str(conf))
	    else:
	    	text = "Unknown Person"
	    # img_item = "my-image.png"
	    # cv2.imwrite(img_item, roi_gray)

	    x_end = x+w
	    y_end = y+h
	    color = (0, 255, 0)
	    stroke = 3
	    cv2.rectangle(frame, (x, y), (x_end, y_end), color, stroke)
	    stroke = 2
	    color = (255, 255, 255)
	    font = cv2.FONT_HERSHEY_SIMPLEX
	    cv2.putText(frame, text, (x, y-10), font, 1, color, stroke, cv2.LINE_AA)
	    
	cv2.imshow('frame', frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
