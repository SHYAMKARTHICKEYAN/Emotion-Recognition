from keras.models import load_model
import numpy as np
import dlib
import cv2
import warnings
warnings.filterwarnings('ignore')

def shapePoints(shape):
        coords = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords

def rectPoints(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)

faceLandmarks = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(faceLandmarks)

emotions = {0: "Angry", 1:"Disgust", 2: "Fear", 3:"Happy", 4:"Sad", 5: "Suprise", 6: "Neutral"}

emotionModelPath = 'emotionModel.hdf5' 
emotionClassifier = load_model(emotionModelPath, compile=False)
emotionTargetSize = emotionClassifier.input_shape[1:3]

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
	ret, frame = cap.read()
	if not ret:
		break

	grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector(grayFrame, 0)
	for rect in rects:
		# print(rect)
		shape = predictor(grayFrame, rect)
		points = shapePoints(shape)
		(x, y, w, h) = rectPoints(rect)
		grayFace = grayFrame[y:y + h, x:x + w]
		try:
			grayFace = cv2.resize(grayFace, (emotionTargetSize))
		except:
			continue
		grayFace = grayFace.astype('float32')
		grayFace = grayFace / 255.0
		grayFace = (grayFace - 0.5) * 2.0
		grayFace = np.expand_dims(grayFace, 0)
		grayFace = np.expand_dims(grayFace, -1)
		emotion_prediction = emotionClassifier.predict(grayFace)
		emotion_probability = np.max(emotion_prediction)

		emotion_label_arg = np.argmax(emotion_prediction)
		# print(emotions[emotion_label_arg])
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

		cv2.putText(frame, emotions[emotion_label_arg], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

	cv2.imshow('Emotion', frame)
	if cv2.waitKey(1) & 0xff == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()