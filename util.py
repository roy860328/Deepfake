import cv2
import numpy as np
import matplotlib.pyplot as plt

class VideoPreprocess():
	"""docstring for VideoPreprocess"""
	def __init__(self, max_n_frames, width, height):
		self.width = width
		self.height = height
		self.max_n_frames = max_n_frames
		
	def get_frames(self, file, print_detail=False):
		print("\n=== Start get_frames ===")
		print(file)
		cap = self._read_video(file)
		if(print_detail):
			self._video_info(cap)
		frames = self._video_frame(cap)
		return frames
	def _read_video(self, file):
		cap = cv2.VideoCapture(file)
		if(not cap.isOpened()):
			raise "file error"
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
			# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame = cv2.resize(frame, (self.width, self.height))
			# print(frame.shape)
			frames.append(frame)
			# frame = cv2.resize(frame, (500, 500))
			
			# frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
			# cv2.imshow('frame', frame)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		if len(frames) < self.max_n_frames:
			for i in range(len(frames), self.max_n_frames):
				frames.append(frames[len(frames)-1])
			print(len(frames))
			# raise "The frames less the 100"
		return frames

LABELS = ['REAL','FAKE']

class DataProcess():
	"""docstring for DataProcess"""
	def __init__(self, df_trains, videoPreprocess):
		print("init")
		self.df_trains = df_trains
		self.videoPreprocess = videoPreprocess
	
	def get_generater_data(self, data_size):
		for df_train in self.df_trains:
			print("\nNew df_train")
			elements = list(df_train.columns.values)
			lower, upper = 0, data_size
			while True:
				training_set = []
				video_labels = []
				for element in elements[lower:upper]:
					frames = self.videoPreprocess.get_frames("train_sample_videos/" + element)
					training_set.append(frames)
					video_labels.append(LABELS.index(df_train[element]['label']))
				training_set = np.asarray(training_set)
				video_labels = np.asarray(video_labels)
				
				lower = upper
				upper += data_size
				if lower == len(list(df_train.columns.values)):
					break
				if upper > len(list(df_train.columns.values)):
					upper = len(list(df_train.columns.values))
				print("lower: ", lower)
				print("upper: ", upper)
				yield [training_set, video_labels]

if __name__ == "__main__":
	file = "train_sample_videos/aagfhgtpmv.mp4"
	print(len(get_frames(file)))