import numpy as np
import argparse
import time
import cv2 as cv

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

net = cv.dnn.readNetFromCaffe(args["prototxt"], args["model"])
vs = cv.VideoCapture(0)

while True:
	ret, frame = vs.read()
	if not ret:
            cv.waitKey()
            break
        
	(h, w) = frame.shape[:2]
	blob = cv.dnn.blobFromImage(frame, 1.0, (w, h))
	net.setInput(blob)
	detections = net.forward()
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence < args["confidence"]:
			continue
			
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
		cv.rectangle(frame, (startX, startY), (endX, endY), 
			     (0, 0, 255), 2)
			     
		t, _ = net.getPerfProfile()
		label = 'Time spent on one frame: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
		cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
		
		label = '%.4f' % (confidence)
		cv.putText(frame, label, (0, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

		cv.imshow('result', frame)
	
		key = cv.waitKey(1) & 0xFF

		if key == ord("q"):
			exit()
		
cv.destroyAllWindows()

