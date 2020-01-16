# from . import util
import util

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing

# from tensorflow.keras.datasets import imdb
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Activation, Embedding

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation, Embedding
from keras.layers import Dropout
from keras.constraints import max_norm
from keras import regularizers

import time


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")



def preprocess():

    train_x = train_df["text"][:-800]
    train_y = train_df["target"][:-800]
    val_x = train_df["text"][7000:7600]
    val_y = train_df["target"][7000:7600]
    test_x = test_df["text"]

    _, val_x, vocabulary, reversed_dictionary = util.load_data_2_embed(train_x, val_x)
    train_x, test_x, vocabulary, reversed_dictionary = util.load_data_2_embed(train_x, test_x)

    print(train_x[0])
    print(train_x[1])
    print(len(train_x[0]))
    print(len(train_x[1]))

    ## preprocess
    train_x = sequence.pad_sequences(train_x, )
    val_x = sequence.pad_sequences(val_x, )
    test_x = sequence.pad_sequences(test_x, )
    print(train_x.shape)
    print(train_x[0])
    print(train_x[0].shape)

    return train_x, train_y, val_x, val_y, test_x, vocabulary

def export(predictions):
    sample_submission["target"] = predictions
    print(sample_submission.head())
    print(sample_submission[0:20])
    sample_submission.to_csv("submission.csv", index=False)

class LSTM_Implement():
    """docstring for LSTM"""
    def __init__(self, vocabulary, hidden_size=15):
        self.vocabulary = vocabulary
        self.hidden_size = hidden_size
        
    def create_model(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocabulary, self.hidden_size, input_length=None))
        self.model.add(Dropout(1))
        self.model.add(LSTM(self.hidden_size, dropout=0.2, recurrent_dropout=0.2, 
        					kernel_constraint=max_norm(1), bias_constraint=max_norm(1), 
        					kernel_regularizer=regularizers.l2(0.01), 
        						))
        self.model.add(Dropout(1))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train(self, train_x, train_y, val_x, val_y):
        # model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(train_x, train_y, epochs=5, batch_size=100, shuffle=True, verbose=2)
        self.eval(val_x, val_y)

    def eval(self, val_x, val_y):
        scores = self.model.evaluate(val_x, val_y, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))

    def predict(self, test_x):
        return self.model.predict_classes(test_x)

if __name__ == "__main__":
    train_x, train_y, val_x, val_y, test_x, vocabulary = preprocess()
    model = LSTM_Implement(vocabulary)
    model.create_model()
    model.train(train_x, train_y, val_x, val_y)
    # model.eval(val_x, val_y)
    model.train(train_x, train_y, val_x, val_y)
    # model.eval(val_x, val_y)

    predictions = model.predict(test_x)

    export(predictions)