#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = "Prashant Shivarm Bhat"
__email__ = "PrashantShivaram@outlook.com"

import tensorflow as tf
import numpy as np
import sys
from util import tokenize_words
from sklearn.base import BaseEstimator, TransformerMixin
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CharacterNLG(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.seq_length = 100
        self.x_data = []
        self.y_data = []
        self.X = None
        self.y = None
        self.model = None
        self.processed_inputs = None
        self.chars = None
        self.vocab_len = None
        self.weights_fname = 'model_weights_saved.hdf5'

    def get_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(256, input_shape=(self.X.shape[1], self.X.shape[2]), return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(256, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(128))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(self.y.shape[1], activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def preprocess(self):
        file = open("../data/data.txt").read()
        self.processed_inputs = tokenize_words(file)
        self.chars = sorted(list(set(self.processed_inputs)))
        char_to_num = dict((c, i) for i, c in enumerate(self.chars))
        input_len = len(self.processed_inputs)
        self.vocab_len = len(self.chars)
        print("Total number of characters:", input_len)
        print("Total vocab:", self.vocab_len)

        for i in range(0, input_len - self.seq_length, 1):
            in_seq = self.processed_inputs[i:i + self.seq_length]
            out_seq = self.processed_inputs[i + self.seq_length]
            self.x_data.append([char_to_num[char] for char in in_seq])
            self.y_data.append(char_to_num[out_seq])

        n_patterns = len(self.x_data)
        self.X = np.reshape(self.x_data, (n_patterns, self.seq_length, 1))
        self.X = self.X / float(self.vocab_len)
        self.y = tf.keras.utils.to_categorical(self.y_data)

    def fit(self):
        self.preprocess()
        self.model = self.get_model()
        checkpoint = tf.keras.callbacks.ModelCheckpoint(self.weights_fname, monitor='loss', verbose=1, save_best_only=True,
                                                        mode='min')
        desired_callbacks = [checkpoint]
        self.model.fit(self.X, self.y, epochs=30, batch_size=256, callbacks=desired_callbacks)


    def random_pattern(self):
        start = np.random.randint(0, len(self.x_data) - 1)
        pattern = self.x_data[start]
        print("Random Seed:")
        num_to_char = dict((i, c) for i, c in enumerate(self.chars))
        print("\"", ''.join([num_to_char[value] for value in pattern]), "\"")
        return pattern

    def generate_text(self):
        self.model.load_weights(self.weights_fname)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        num_to_char = dict((i, c) for i, c in enumerate(self.chars))
        pattern = self.random_pattern()
        tex = ""
        for i in range(100):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(self.vocab_len)
            prediction = self.model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = num_to_char[index]
            seq_in = [num_to_char[value] for value in pattern]
            print(result)
            tex += result
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        return tex


char = CharacterNLG()
char.fit()
text = char.generate_text()
print()
