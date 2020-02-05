from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, BatchNormalization, Activation, Conv3D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalAveragePooling3D, MaxPooling3D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Nadam, Adamax, RMSprop
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.models import load_model
from keras.utils import multi_gpu_model

import time

# # ### prevent
import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
#                                     # (nothing gets printed in Jupyter, only if you run it standalone)
# session = tf.InteractiveSession(config=config)
# # set_session(sess)  # set this TensorFlow session as the default session for Keras
# K.set_session(session)

## 
with K.tf.device('/gpu:0'):
	config = tf.ConfigProto(intra_op_parallelism_threads=4,\
		   inter_op_parallelism_threads=4, allow_soft_placement=True,\
		   device_count = {'CPU' : 1, 'GPU' : 2})
	config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
	config.log_device_placement = True  # to log device placement (on which device the operation ran)
	session = tf.Session(config=config)
	K.set_session(session)

## resnet3d
# https://github.com/JihongJu/keras-resnet3d/blob/master/resnet3d/resnet3d.py
class CNNImplement():
	"""docstring for LSTM"""
	def __init__(self):
		pass
		
	def create_Sequential_model(self, frames, rows, columns, channels, classification):
		input_shape = self.check_input_format(rows, columns)	# input_shape = (rows, columns, 3)
		video = Input(shape=(input_shape[0],
							 input_shape[1],
							 input_shape[2]))
		self.model = Sequential()
		a = 128
		# a = 32
		self.model.add(Conv3D(a, 7, strides=(1,1,1), padding='same', activation='relu', 
															   input_shape=(frames,
																		input_shape[0],
																		input_shape[1],
																		input_shape[2])))
		self.model.add(Conv3D(64, 5, strides=(1,1,1), padding='same', activation='relu'))
		self.model.add(Conv3D(32, 3, strides=(1,1,1), padding='same', activation='relu'))
		# self.model.add(MaxPooling3D())
		self.model.add(GlobalAveragePooling3D())
		self.model.add(Dense(output_dim=512, activation="relu"))
		self.model.add(Dense(output_dim=classification, activation="sigmoid"))
		# self.model.add(Dense(1, activation="sigmoid"))
		optimizer = Nadam(lr=0.002,
						  beta_1=0.9,
						  beta_2=0.999,
						  epsilon=1e-08,
						  schedule_decay=0.004)
		optimizer = Adamax(clipnorm=1.)
		# self.model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy', self.custom_loss_function]) 
		self.model.compile(optimizer='adam', loss=self.custom_loss_function, metrics=['accuracy', self.custom_loss_function]) 
		self.model.summary()
	# ResNet50
	def create_ResNet50_model(self, frames, rows, columns, channels, classification):
		input_shape = self.check_input_format(rows, columns)	# input_shape = (rows, columns, 3)
		video = Input(shape=(frames,
							 input_shape[0],
							 input_shape[1],
							 input_shape[2]))
		cnn_base = ResNet50(include_top=False, 
							weights='imagenet', 
							input_tensor=None, 
							input_shape=input_shape, 
							pooling=None, 
							classes=1)

		cnn_out = GlobalAveragePooling2D()(cnn_base.output)
		cnn = Model(input=cnn_base.input, output=cnn_out)
		cnn.trainable = True
		encoded_frames = TimeDistributed(cnn)(video)
		encoded_sequence = LSTM(128)(encoded_frames)
		# encoded_sequence = LSTM(256, return_sequences=False)(cnn_out)
		hidden_layer = Dense(output_dim=512, activation="relu")(encoded_sequence)
		outputs = Dense(output_dim=classification, activation="sigmoid")(hidden_layer)
		self.model = Model([video], outputs)
		self.gpu_model = multi_gpu_model(self.model, gpus=2)
		optimizer = Nadam(lr=0.002,
						  beta_1=0.9,
						  beta_2=0.999,
						  epsilon=1e-08,
						  schedule_decay=0.004)
		self.gpu_model.compile(optimizer='adam', loss=self.custom_loss_function, metrics=['accuracy', self.custom_loss_function]) 
		self.gpu_model.summary()
	def create_model(self, frames, rows, columns, channels, classification):
		input_shape_ori = self.check_input_format(rows, columns)	# input_shape = (rows, columns, 3)
		video = Input(shape=(frames,
							 input_shape_ori[0],
							 input_shape_ori[1],
							 input_shape_ori[2]))

		input_shape = Input(shape=(input_shape_ori))
		conv_x = BatchNormalization()(input_shape)
		conv_x = Conv2D(128, (7,1), padding='same')(conv_x)
		conv_x = BatchNormalization()(conv_x)
		conv_x = Activation('relu')(conv_x)

		conv_x = Conv2D(64, (5,1), padding='same')(conv_x)
		conv_x = BatchNormalization()(conv_x)
		conv_x = Activation('relu')(conv_x)
		
		conv_x = Conv2D(32, (3,1), padding='same')(conv_x)
			
		cnn_out = GlobalAveragePooling2D()(conv_x)
		cnn = Model(input=input_shape, output=cnn_out)
		cnn.trainable = True
		encoded_frames = TimeDistributed(cnn)(video)
		encoded_sequence = LSTM(128)(encoded_frames)
		# encoded_sequence = LSTM(256, return_sequences=False)(cnn_out)
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

	def custom_loss_function(self, y_real, y_predict):
		# import math 
		# custom_loss = y_real * math.log(y_predict) + (1-y_real) * math.log((1-y_predict))
		# y_real = K.print_tensor(y_real, message='y_real = ')
		y_predict = K.print_tensor(y_predict, message='y_predict = ')

		y_predict = K.clip(y_predict, 1e-3, 1 - 1e-3)
		custom_loss = -K.mean( y_real*K.log(y_predict) + (1-y_real)*K.log(1-y_predict) )
		# custom_loss = -( K.log(y_real-y_predict))
		# custom_loss = binary_crossentropy(y_real, y_predict)
		# custom_loss = K.mean( (y_real-y_predict)**2 )

		# custom_loss = K.print_tensor(custom_loss, message='custom_loss = ')
		return custom_loss

	def train(self, train_x, train_y, epochs, batch_size, tensorboard_callback):
		self.gpu_model.fit(train_x, train_y, 
						epochs=epochs, 
						batch_size=batch_size, 
						verbose=2, 
						callbacks=[tensorboard_callback])

	def train_generater(self, generater, steps_per_epoch, epochs, tensorboard_callback):
		self.gpu_model.fit_generator(generater, 
								 steps_per_epoch=steps_per_epoch, 
								 epochs=epochs, 
								 verbose=2, 
								 callbacks=[tensorboard_callback], 
								 shuffle=True,
								 max_queue_size=2)

	def eval(self, val_x, val_y):
		scores = self.gpu_model.evaluate(val_x, val_y, verbose=0)
		# print("Accuracy: %.2f%%" % (scores[1]*100))

	def predict(self, test_x):
		return str(self.gpu_model.predict(test_x)).replace("[", "").replace("]", "")

	def load_model(self, path):
		# self.model = load_model(path + ".h5", custom_objects={'custom_loss_function': self.custom_loss_function})
		self.model.load_weights(path + ".h5")
	def save_model(self, path):
		self.model.save(path + ".h5")


if __name__ == "__main__":
	train_x, train_y, val_x, val_y, test_x, vocabulary = preprocess()
	model = LSTM_Implement(vocabulary)
	model.create_model()
	model.train(train_x, train_y)
	model.eval(val_x, val_y)
	predictions = model.predict(test_x)

	export(predictions)
