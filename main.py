import pandas as pd
import numpy as np

import util
import CNN

	
### Setting
data_size = 1
batch_size = 1
if batch_size > data_size:
	batch_size = data_size
test = False
sample_submission = pd.read_csv("sample_submission.csv")
sample_submission_json = util.processs_data_to_json(sample_submission)

### Preprocess 
videoPreprocess = util.VideoPreprocess(max_n_frames=100, width=224, height=224)
#
train_dataProcess = util.get_data_class('train_sample_videos/metadata.json', path="train_sample_videos/", videoPreprocess=videoPreprocess)
training_set = train_dataProcess.get_generater_data(data_size=data_size)
test_dataProcess = util.get_data_class(sample_submission_json, path="test_videos/", videoPreprocess=videoPreprocess)
testing_set = test_dataProcess.get_generater_data(data_size=data_size)



def test_print_data():
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

	if test:
		test_print_data()
	else:
		model = CNN.CNNImplement()
		model.create_model( frames=videoPreprocess.max_n_frames, 
							rows=videoPreprocess.width, 
							columns=videoPreprocess.height, 
							channels=3,
							classification=1)
		for _ in range(1):
			for x, y in training_set:
				# print(x.shape)
				# print(y)
				model.train(train_x=x, train_y=y, epochs=1, batch_size=batch_size)
				# break

		predict_label = []
		for x, y in testing_set:
			result = model.predict(test_x=x)
			print(result)
			predict_label.append(str(result).replace("[", "").replace("]", ""))

		util.export(sample_submission, predict_label)