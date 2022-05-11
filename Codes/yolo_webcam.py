import numpy as np
import cv2
import os
from datetime import datetime
from face_recognition_mysql import *
from time import sleep


LABELS_FILE=r"C:\Users\mehrh\program\FinalProject\darknet\data\coco.names"
CONFIG_FILE=r"C:\Users\mehrh\program\FinalProject\darknet\cfg\yolov3.cfg"
WEIGHTS_FILE=r"C:\Users\mehrh\program\FinalProject\darknet\cfg\yolov3.weights"
CONFIDENCE_THRESHOLD=0.3

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

WAIT_SECS = 1

net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

vs = cv2.VideoCapture(0)
(W, H) = (None, None)

def find_object(layerOutputs, frame):
	boxes = []
	confidences = []
	classIDs = []
	
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if confidence > CONFIDENCE_THRESHOLD:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, CONFIDENCE_THRESHOLD)
	
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv2.putText(frame, str(datetime.now()), (20, 40),
				cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2, cv2.LINE_AA)
	
			encode_mysql(frame)

while True:
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	
	net.setInput(blob)
	ln = net.getUnconnectedOutLayersNames()
	layerOutputs = net.forward(ln)

	find_object(layerOutputs, frame)
	sleep(WAIT_SECS)

	cv2.imshow('WebCam', frame)
	
	if cv2.waitKey(1) == ord('q'):
		break

vs.release()
cv2.destroyAllWindows()
