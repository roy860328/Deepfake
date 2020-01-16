from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam
from keras import backend as K


LABELS = ['REAL','FAKE']

class CNNImplement():
	"""docstring for LSTM"""
	def __init__(self):
		pass
		
	def create_model(self, frames, rows, columns, channels, classification):
		input_shape = self.check_input_format(rows, columns)	# input_shape = (rows, columns, 3)
		video = Input(shape=(frames,
							 input_shape[0],
							 input_shape[1],
							 input_shape[2]))
		cnn_base = VGG16(input_shape=input_shape,
						 weights="imagenet",
						 include_top=False)
		cnn_out = GlobalAveragePooling2D()(cnn_base.output)
		cnn = Model(input=cnn_base.input, output=cnn_out)
		cnn.trainable = True
		encoded_frames = TimeDistributed(cnn)(video)
		encoded_sequence = LSTM(256)(encoded_frames)
		hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_sequence)
		outputs = Dense(output_dim=classification, activation="sigmoid")(hidden_layer)
		self.model = Model([video], outputs)
		optimizer = Nadam(lr=0.002,
						  beta_1=0.9,
						  beta_2=0.999,
						  epsilon=1e-08,
						  schedule_decay=0.004)
		self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
		self.model.summary()
	## https://riptutorial.com/keras/example/29812/vgg-16-cnn-and-lstm-for-video-classification
	def ori_create_model(self):
		video = Input(shape=(frames,
					 channels,
					 rows,
					 columns))
		cnn_base = VGG16(input_shape=(channels,
									  rows,
									  columns),
						 weights="imagenet",
						 include_top=False)
		cnn_out = GlobalAveragePooling2D()(cnn_base.output)
		cnn = Model(input=cnn_base.input, output=cnn_out)
		cnn.trainable = False
		encoded_frames = TimeDistributed(cnn)(video)
		encoded_sequence = LSTM(256)(encoded_frames)
		hidden_layer = Dense(output_dim=1024, activation="relu")(encoded_sequence)
		outputs = Dense(output_dim=classes, activation="softmax")(hidden_layer)
		model = Model([video], outputs)
		optimizer = Nadam(lr=0.002,
						  beta_1=0.9,
						  beta_2=0.999,
						  epsilon=1e-08,
						  schedule_decay=0.004)
		model.compile(loss="categorical_crossentropy",
					  optimizer=optimizer,
					  metrics=["categorical_accuracy"]) 

	def check_input_format(self, rows, columns):
		if K.image_data_format() == 'channels_first':
			# input_crop = input_crop.reshape(input_crop.shape[0], 3, rows, columns)
			input_shape = (3, rows, columns)
		else:
			# input_crop = input_crop.reshape(input_crop.shape[0], rows, columns, 3)
			input_shape = (rows, columns, 3)

		# input_crop = Input(shape=input_shape)
		# return input_crop
		return input_shape

	def train(self, train_x, train_y, epochs, batch_size):
		# model.compile(loss='mean_squared_error', optimizer='adam')
		self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2)

	def eval(self, val_x, val_y):
		scores = self.model.evaluate(val_x, val_y, verbose=0)
		# print("Accuracy: %.2f%%" % (scores[1]*100))

	def predict(self, test_x):
		return str(self.model.predict(test_x)).replace("[", "").replace("]", "")

if __name__ == "__main__":
	train_x, train_y, val_x, val_y, test_x, vocabulary = preprocess()
	model = LSTM_Implement(vocabulary)
	model.create_model()
	model.train(train_x, train_y)
	model.eval(val_x, val_y)
	predictions = model.predict(test_x)

	export(predictions)
