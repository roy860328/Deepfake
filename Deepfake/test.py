import cv2
import pickle
from PIL import Image
# import face_recognition
import os
import numpy as np
# from mtcnn import MTCNN

def load_pickle(filename):
	file = open(filename, 'rb')
	# dump information to that file
	data = pickle.load(file)
	# close the file
	file.close()
	return data

def read_pickle(file):
	frames = load_pickle(file)
	for img in frames:
		cv2.imshow('image',img)
		cv2.waitKey(0)

def face_detect():
	img = cv2.imread("123.png")
	face_locations = face_recognition.face_locations(img)
	print(face_locations)
	top, right, bottom, left = face_locations[0]
	faceImage = img[top:bottom, left:right]
	final = Image.fromarray(faceImage)
	final.save("img%s.png" % (str("0")), "PNG")

def face_detect_MTCNN():
	img = cv2.cvtColor(cv2.imread("123.png"), cv2.COLOR_BGR2RGB)
	detector = MTCNN()
	faces = detector.detect_faces(img)
	x, y, width, height = faces[0]["box"]
	faceImage = img[y:y+height, x:x+width]
	faceImage = cv2.cvtColor(faceImage, cv2.COLOR_RGB2BGR)
	cv2.imshow('image',faceImage)
	cv2.waitKey(0)

def concatenate_video():
	cap = cv2.VideoCapture('out.mp4',0)
	cap1 = cv2.VideoCapture('601200231.176403.mp4',0)

	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	out = cv2.VideoWriter('output.mp4', fourcc, 
							cap.get(cv2.CAP_PROP_FPS), 
							(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

	while(cap.isOpened()):

		ret, frame = cap.read()
		ret1, frame1 = cap1.read()
		if ret == True and ret1 == True: 
			both = np.concatenate((frame, frame1), axis=1)
			# cv2.imshow('Frame', both)
			out.write(both)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else: 
			break

	cap.release()
	cap1.release()
	out.release()

	# cv2.waitKey(0)
	cv2.destroyAllWindows()

def resize_img(path, width, height):
	for file_name in os.listdir(path):
		if file_name.find(".jpg") == -1:
			continue
		print("\nFile: "+ file_name)
		img = cv2.imread(path + file_name)
		img = cv2.resize(img, (width, height))

		cv2.imwrite(path + file_name, img)

def resize_video(path, width, height):
	cap = cv2.VideoCapture(path,0)

	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	out = cv2.VideoWriter("d_" + path, fourcc, 
							cap.get(cv2.CAP_PROP_FPS), 
							(width, height))
	# out = cv2.VideoWriter('resizeviode.mp4', fourcc, 
	# 						cap.get(cv2.CAP_PROP_FPS), 
	# 						(width, height))
	while(cap.isOpened()):

		ret, frame = cap.read()
		if ret == True: 
			frame = cv2.resize(frame, (width, height))
			out.write(frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else: 
			break
	cap.release()
	out.release()
	cv2.destroyAllWindows()

# if os.path.isdir("test_videos/"):
# 	print("isdir")
# 	os.makedirs('eye\\')

# concatenate_video()
read_pickle("zxprilbsxp.pickle")
# face_detect_MTCNN()
# resize_img("./", 640, 480)
# resize_video("601200230.656149.mp4", 640, 480)