import numpy as np
import cv2
import dlib
import mysql.connector
from PIL import Image
import os
from datetime import datetime

LABELS_FILE=r"C:\Users\mehrh\program\FinalProject\darknet\data\coco.names"
CONFIG_FILE=r"C:\Users\mehrh\program\FinalProject\darknet\cfg\yolov3.cfg"
WEIGHTS_FILE=r"C:\Users\mehrh\program\FinalProject\darknet\cfg\yolov3.weights"
CONFIDENCE_THRESHOLD=0.3

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='face_recognition_mapsa')
cursor=cnx.cursor()
# mysql_table = 'webcam'


net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(
	r'C:\Users\mehrh\program\FinalProject\face_recognition\shape_predictor_68_face_landmarks.dat'
)
face_recognition_model = dlib.face_recognition_model_v1(
	r'C:\Users\mehrh\program\FinalProject\face_recognition\dlib_face_recognition_resnet_model_v1.dat'
)
TOLERANCE = 0.6

def get_face_encodings(frame):
    image = np.asarray(frame)
    
    detected_faces = face_detector(image, 1)
    shapes_faces = [shape_predictor(image, face) for face in detected_faces]
    
    return [np.array(face_recognition_model.compute_face_descriptor(image, face_pose, 1)) for face_pose in shapes_faces]

def compare_face_encodings(known_encoded, face):

    return (np.linalg.norm(known_encoded - face, axis=1) <= TOLERANCE)

def find_match(known_encoded, frame_encoded):
	matches = compare_face_encodings(known_encoded, frame_encoded)

	if not any(matches):
		cursor.execute(
			'INSERT INTO webcam(encoded_image, repeated_number) VALUES (\'%s\',\'%i\')' %(str(frame_encoded.tolist()), 0)
		)
		cnx.commit()
		return 

	else:
		count = 1
		for match in matches:
			if match:
				cursor.execute(
					'UPDATE webcam SET repeated_number = repeated_number + 1 WHERE face_id = %i' %(count)
				)
				cnx.commit()
				return
			count += 1
	

def encode_mysql(frame):
	frame_encoded = get_face_encodings(frame)
	face_encoded_retrieve = []
	
	if len(frame_encoded) != 1:
		return
	else:
		cursor.execute('SELECT encoded_image FROM webcam;')

		for item in cursor:
			face_encoded_retrieve.append(np.asarray(eval(item[0])))
		
		if not face_encoded_retrieve:
			cursor.execute(
				'INSERT INTO webcam (encoded_image, repeated_number) VALUES (\'%s\',\'%i\')' %(str(frame_encoded[0].tolist()), 0)
			)
			cnx.commit()
		
	find_match(face_encoded_retrieve, frame_encoded[0])


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
	
	cv2.imshow('WebCam', frame)
	
	if cv2.waitKey(1) == ord('q'):
		break

print("[INFO] cleaning up...")
vs.release()
cv2.destroyAllWindows()
