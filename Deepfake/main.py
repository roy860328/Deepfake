import pandas as pd
import numpy as np
from datetime import datetime

import util
import CNN
import keras

### Setting
data_size = 1
batch_size = 1
max_n_frames = 75
if batch_size > data_size:
	batch_size = data_size
# number_train_data always < real train data
number_train_data = 10	#1700
print(number_train_data)
test = False

### Preprocess 
videoPreprocess = util.VideoPreprocess(max_n_frames=max_n_frames, width=224, height=224)

# train data
path = ["train_sample_videos/"]#, "train_00/"]
json_path = [p + "metadata.json" for p in path]
train_dataProcess = util.get_data_class(json_path, path=path, videoPreprocess=videoPreprocess)
training_set = train_dataProcess.get_generater_data(data_size=data_size)

# test data
sample_submission = pd.read_csv("sample_submission.csv")
sample_submission_json = util.processs_data_to_json(sample_submission)
path = ["test_videos/"]
test_dataProcess = util.get_data_class([sample_submission_json], path=path, videoPreprocess=videoPreprocess)
testing_set = test_dataProcess.get_generater_data(data_size=data_size)


## Tensorboard
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
# tensorboard_callback = keras.callbacks.tensorboard_v1.TensorBoard(log_dir=logdir)


def test_print_data():
	# train_dataProcess.shuffle_data()
	# print(train_dataProcess.df_trains)
	# train_dataProcess.shuffle_data()
	# print(train_dataProcess.df_trains)

	for x, y in training_set:
		print(x.shape)
		print(y)
		break
	predict_label = []
	for x, y in testing_set:
		print(x.shape)
		predict_label.append("1")
		break
	util.export(sample_submission, predict_label)

if __name__ == "__main__":

	model = CNN.CNNImplement()
	model.create_ResNet50_model( frames=videoPreprocess.max_n_frames, 
						rows=videoPreprocess.width, 
						columns=videoPreprocess.height, 
						channels=3,
						classification=1)
	if test:
		# test_print_data()
		print("\n\n\n\n\n\n\n\n\n\n\n=======  load  =======")
		model.load_model("logs/scalars/" + "20200128-065253_ResNet50_model")

		predict_label = []
		for x, y in testing_set:
			result = model.predict(test_x=x)
			print("        test: ",result)
			predict_label.append(result)

		util.export(sample_submission, predict_label)

	else:

		for _ in range(2000):
			print("======\n\n\n\n")
			print(_)
			model.train_generater(training_set, steps_per_epoch=number_train_data/batch_size, epochs=1, tensorboard_callback=tensorboard_callback)
			train_dataProcess.shuffle_data()
			training_set = train_dataProcess.get_generater_data(data_size=data_size)

		model.save_model(logdir + "_ResNet50_model")

		predict_label = []
		for x, y in testing_set:
			result = model.predict(test_x=x)
			print("        test: ",result)
			predict_label.append(result)

		util.export(sample_submission, predict_label)