import pandas as pd
import numpy as np

import util
import CNN

data_size = 3
batch_size = 3
if batch_size > data_size:
	batch_size = data_size


def get_data_class(videoPreprocess):
	df_train = pd.read_json('train_sample_videos/metadata.json')
	df_trains = [df_train]

	dataProcess = util.DataProcess(df_trains, videoPreprocess)
	return dataProcess

if __name__ == "__main__":

	videoPreprocess = util.VideoPreprocess(max_n_frames=100, width=540, height=960)
	videoPreprocess = util.VideoPreprocess(max_n_frames=100, width=224, height=224)

	dataProcess = get_data_class(videoPreprocess)
	training_set = dataProcess.get_generater_data(data_size=data_size)

	model = CNN.CNNImplement()
	model.create_model( frames=videoPreprocess.max_n_frames, 
						rows=videoPreprocess.width, 
						columns=videoPreprocess.height, 
						channels=3,
						classification=1)

	for x, y in training_set:
		print(x.shape)
		print(y)
		model.train(train_x=x, train_y=y, epochs=1, batch_size=batch_size)

	training_set = dataProcess.get_generater_data(data_size=20)
	for x, y in training_set:
		model.eval(val_x=x, val_y=y)