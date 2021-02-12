# USAGE
# python detect_faces.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

# Библиотека imutils позволяет повторить те же действия, что и OpenCV в плане
# открытия видео-потока и смены разрешения картинки, однако в данном случае,
# она будет игнорировать "лишние" кадры, если компьютер не успел их обработать
# (не кладутся изображения в буфер)
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

# Блок управления роботом
#############################
import rospy
import roslib
from promobot_msgs.msg import FaceRect
from promobot_msgs.msg import FaceRectArray
from promobot_msgs.msg import PowerValue
#############################

# Подставляем файлы в виде аргументов
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Подгружаем модель
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Инициализируем видео-поток
print("[INFO] starting video stream...")
vs = VideoStream("rtsp://admin:admin@192.168.250.99:554/ch01/0").start()
time.sleep(2.0)

# Функции по управлению роботом
#############################
rospy.init_node("face_detect")

pub_rect = rospy.Publisher('/face/rect/array', FaceRectArray, queue_size=20, latch=True)

#############################


# Обработка каждого кадра находится в функции
def spinOnce():

	# Захватываем кадр и делаем смену разрешения
	frame = vs.read()
	frame = imutils.resize(frame, width=300)
 
	# Конвертируем в 4-х мерный массив
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# Запускаем блок предсказания
	net.setInput(blob)
	detections = net.forward()

	# Цикл на обработку детекций
	for i in range(0, detections.shape[2]):
	
		# Достаем значение по индексу 2, где хранится оценка
		# "уверенности" нейронной сети
		confidence = detections[0, 0, i, 2]

		# Убираем лишние детекции, оценка которых не превышает порог
		if confidence < args["confidence"]:
			continue

		# Забираем нормализованные координаты детекций и умножаем на разрешение
		# нашего изображения
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")
 
		# Рисуем AABB (bounding box)
		text = "{:.2f}%".format(confidence * 100)
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		# Блок передачи координат найденных детекций
		###############################################
		msg_r = FaceRect()
		msg_face_rects = FaceRectArray()
		msg_r.track_id    = 0
		msg_r.is_local    = True
		msg_r.is_tracking = True
		msg_r.x           = startX / 300.0
		msg_r.y           = startY / 300.0
		msg_r.width       = abs(startX-endX) / 300.0
		msg_r.height      = abs(startY-endY) / 300.0
		msg_face_rects.rects.append( msg_r )
		pub_rect.publish(msg_face_rects)
		###############################################


	# Показыавем результат
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		exit()

while not rospy.is_shutdown():
    spinOnce()

cv2.destroyAllWindows()
vs.stop()
