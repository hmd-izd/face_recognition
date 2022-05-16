import numpy as np
import dlib
import mysql.connector
from PIL import Image
from datetime import datetime




cnx = mysql.connector.connect(user='root', password='',
                              host='127.0.0.1',
                              database='face_recognition_mapsa')
cursor=cnx.cursor()
# mysql_table = 'webcam'

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(
	r'C:\Users\mehrh\program\FinalProject\face_recognition\shape_predictor_68_face_landmarks.dat'
)
face_recognition_model = dlib.face_recognition_model_v1(
	r'C:\Users\mehrh\program\FinalProject\face_recognition\dlib_face_recognition_resnet_model_v1.dat'
)
TOLERANCE = 0.6

moment_person = []

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
	
	global moment_person
	
	cursor.execute('SELECT encoded_image FROM webcam;')

	for item in cursor:
		face_encoded_retrieve.append(np.asarray(eval(item[0])))

	if len(frame_encoded) == 0:
		moment_person = []
		return
	
	else:	
		if not face_encoded_retrieve:
			for iter in range(len(frame_encoded)):
				cursor.execute(
					'INSERT INTO webcam (encoded_image, repeated_number) VALUES (\'%s\',\'%i\')' %(str(frame_encoded[iter].tolist()), 0)
				)
				cnx.commit()
				moment_person.append(frame_encoded[iter])
			return
		
		if not moment_person:
			moment_person = frame_encoded.copy()
			for iter in range(len(frame_encoded)):
				find_match (face_encoded_retrieve, frame_encoded[iter])
		else:
			temp = []
			for iter in range(len(frame_encoded)):
				matches = compare_face_encodings(moment_person, frame_encoded[iter])
				temp.append(frame_encoded[iter])
				if not any(matches):
					find_match(face_encoded_retrieve, frame_encoded[iter])
	
				# else:			
				# 	count = 0
				# 	for match in matches:
				# 		if match:
				# 			temp.append(moment_person[count])
				# 			# return
				# 		count += 1
			moment_person = temp.copy()
			
			# for iter in range(len(moment_person)):
			# 	if moment_person[iter] in frame_encoded:
			# 		temp.append(moment_person[iter])
			# moment_person = temp.copy()

			# moment_person = temp.copy()
			# moment_person = list(set(temp) & set(moment_person))

