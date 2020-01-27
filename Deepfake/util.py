import cv2
import numpy as np
import pandas as pd
import json
import pickle
import os

import matplotlib.pyplot as plt
from PIL import Image
from mtcnn import MTCNN

def get_data_class(json_path, path, videoPreprocess):
	df_trains = []
	for j_p in json_path:
		df_train = pd.read_json(j_p)
		print("\nget_data_class")
		print(df_train)
		print(df_train.isnull().any())
		df_trains.append(df_train)

	dataProcess = DataProcess(df_trains, path, videoPreprocess)
	return dataProcess

def processs_data_to_json(sample_submission):
	sample_submission = sample_submission.to_json(orient='values')
	sample_submission = json.loads(sample_submission)
	sample_submission = {i[0]:{"label":"FAKE"} for i in sample_submission}
	sample_submission = json.dumps(sample_submission)
	return sample_submission

def export(sample_submission, predictions):
	if len(predictions) < len(sample_submission["filename"]):
		[predictions.append("0.5") for _ in range(len(sample_submission["filename"])-len(predictions))]
	sample_submission["label"] = predictions
	print(sample_submission[0:20])
	sample_submission.to_csv("submission.csv", index=False)


class VideoPreprocess():
	"""docstring for VideoPreprocess"""
	def __init__(self, max_n_frames, width, height):
		self.width = width
		self.height = height
		self.max_n_frames = max_n_frames
		self.detector = MTCNN()
		
	def get_frames(self, file, print_detail=False):
		cap = self._read_video(file)
		if(print_detail):
			self._video_info(cap)
		frames = self._video_frame(cap)
		return frames

	# def get_face_frames(self, filename):
	# 	face_frames = self.load_pickle(filename)
	# 	return face_frames

	def _read_video(self, file):
		cap = cv2.VideoCapture(file)
		if(not cap.isOpened()):
			print("file didn't exist: ", file)
			raise "file didn't exist."
		return cap

	def _video_info(self, cap):
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps    = cap.get(cv2.CAP_PROP_FPS)
		print('Total frames: ' + str(total_frames))
		print('width: ' + str(width))
		print('height: ' + str(height))
		print('fps: ' + str(fps))

	def _video_frame(self, cap):
		frames = []
		while True:
			ret, frame = cap.read()
			ret, frame = cap.read()
			ret, frame = cap.read()
			if(ret == False or len(frames)>self.max_n_frames):
				break

			ret, frame = self._face_detect(frame)
			if not (ret):
				continue
			frame = cv2.resize(frame, (self.width, self.height))

			frames.append(frame)

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		cap.release()		
		if len(frames) < self.max_n_frames:
			if(len(frames) == 0):
				return frames
			for i in range(len(frames), self.max_n_frames):
				frames.append(frames[len(frames)-1])
			print(len(frames))
			# raise "The frames less the 100"
		return frames
	def _face_detect(self, frame):
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		face_locations = self.detector.detect_faces(frame)
		if (len(face_locations) == 0):
			print("Face error")
			return False, None
		x, y, width, height = face_locations[0]["box"]
		faceImage = frame[y:y+height, x:x+width]
		cropped = faceImage
		if (cropped.size == 0):
			print("Face error")
			return False, cropped
		cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
		return True, cropped

	def dump_face_detect_img(self, filename, data):
		file = open(filename, 'wb')
		# dump information to that file
		pickle.dump(data, file)
		# close the file
		file.close()
	def load_pickle(self, filename):
		file = open(filename, 'rb')
		# dump information to that file
		data = pickle.load(file)
		# close the file
		file.close()
		return data
	def check_pickle(self, filename):
		if os.path.isfile(filename):
			return True
		else:
			return False
	def save_image(self, name, frame):
		cv2.imwrite(name + '.png', frame)

LABELS = ['REAL','FAKE']

class DataProcess():
	"""docstring for DataProcess"""
	def __init__(self, df_trains, path, videoPreprocess):
		# print("init")
		self.df_trains = df_trains
		self.path = path
		print(self.path)
		self.videoPreprocess = videoPreprocess
	def generate_face_images(self):
		for index, df_train in enumerate(self.df_trains):
			elements = list(df_train.columns.values)
			for element in elements:
				print("=== one element ===")
				print(self.path[0] + element)
				frames = self.videoPreprocess.get_frames(self.path[index] + element)
	def generate_face_frames_pickle(self):
		for index, df_train in enumerate(self.df_trains):
			elements = list(df_train.columns.values)
			for element in elements:
				pickle_file_path = self._get_pickle_file_name(self.path[index], element)
				print(pickle_file_path)
				if self.videoPreprocess.check_pickle(pickle_file_path):
					continue
				print("=== one element ===")
				print(self.path[0] + element)
				frames = self.videoPreprocess.get_frames(self.path[index] + element)
				if (len(frames) == 0):
					continue
				self.videoPreprocess.dump_face_detect_img(pickle_file_path, frames)

	def get_generater_data(self, data_size):
		for index, df_train in enumerate(self.df_trains):
			print("\nNew df_train")
			elements = list(df_train.columns.values)
			lower, upper = 0, data_size
			while True:
				print("===")
				training_set = []
				video_labels = []
				for element in elements[lower:upper]:
					print(self.path[index])
					print(element)
					pickle_file_path = self._get_pickle_file_name(self.path[index], element)
					print(pickle_file_path)
					if not os.path.isfile(pickle_file_path):
						continue
					frames = self.videoPreprocess.load_pickle(pickle_file_path)
					# frames = self.videoPreprocess.get_frames(self.path[index]+element)
					training_set.append(frames[0:self.videoPreprocess.max_n_frames])
					video_labels.append(LABELS.index(df_train[element]['label']))
					# video_labels.append(int(df_train[element]['label']))
				lower = upper
				upper += data_size
				if lower == len(list(df_train.columns.values)):
					print("break")
					break
				if upper > len(list(df_train.columns.values)):
					upper = len(list(df_train.columns.values))
				if (len(training_set) == 0): 
					print("pickle: none")
					continue
				training_set = np.asarray(training_set)
				video_labels = np.asarray(video_labels)
				
				print("        video_labels: ", video_labels, end = '')
				print("        lower: ", lower)
				# print("upper: ", upper)
				yield [training_set, video_labels]
	def shuffle_data(self):
		for index, df_train in enumerate(self.df_trains): 
			self.df_trains[index] = self.df_trains[index].sample(frac=1, axis=1)

	def _get_pickle_file_name(self, path, element):
		return path + element.replace(".mp4", "") + ".pickle"

if __name__ == "__main__":
	videoPreprocess = VideoPreprocess(max_n_frames=100, width=224, height=224)
	#
	train_dataProcess = get_data_class(['train_00/metadata.json'], path=["train_00/"], videoPreprocess=videoPreprocess)
	train_dataProcess.generate_face_frames_pickle()
	# train_dataProcess.generate_face_images()

	# sample_submission = pd.read_csv("sample_submission.csv")
	# sample_submission_json = processs_data_to_json(sample_submission)
	# test_dataProcess = get_data_class(sample_submission_json, path=["test_videos/"], videoPreprocess=videoPreprocess)
	# test_dataProcess.generate_face_frames_pickle()