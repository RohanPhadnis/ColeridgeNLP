'''
Semantic
Analysis
Machine
Understanding
English
Literature
'''

import os
import time
import random
import json
import numpy
from tensorflow import keras

RATIO = 0.9
VOCAB_SIZE = 5000
TIME = time.time()
DIR = '../models/classifier{}'.format(TIME)
os.mkdir(DIR)
encoder = json.JSONEncoder()

with open('../data/rime.txt') as file:
    text = file.read()

sentences = text.split('\n')
sentences = [s for s in sentences if len(s) > 0]
tokenizer = keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)

with open('../data/symbols/sun.txt') as sun, open('../data/symbols/moon.txt') as moon:
    sun_text = sun.read()
    moon_text = moon.read()

master = []
for s in sentences:
    if s in sun_text:
        master.append([0, tokenizer.texts_to_sequences([s])[0]])
    else:
        master.append([1, tokenizer.texts_to_sequences([s])[0]])
random.shuffle(master)
INPUT_LEN = max([len(s[1]) for s in master])
train = master[:int(len(master) * RATIO)]
train_xs = [t[1] for t in train]
train_ys = [t[0] for t in train]
val = master[int(len(master) * RATIO):]
val_xs = [v[1] for v in val]
val_ys = [v[0] for v in val]
train_xs = keras.preprocessing.sequence.pad_sequences(train_xs, maxlen=INPUT_LEN)
train_ys = numpy.array(train_ys)
val_xs = keras.preprocessing.sequence.pad_sequences(val_xs, maxlen=INPUT_LEN)
val_ys = numpy.array(val_ys)

print(train_xs)
print(train_ys)

tb = keras.callbacks.TensorBoard(log_dir='{}/logs'.format(DIR))
model = keras.models.Sequential([
    keras.layers.Embedding(input_dim=VOCAB_SIZE, input_length=INPUT_LEN, output_dim=16),
    keras.layers.Bidirectional(layer=keras.layers.LSTM(units=16, return_sequences=True)),
    keras.layers.Bidirectional(layer=keras.layers.LSTM(units=32)),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()
model.fit(x=train_xs,
          y=train_ys,
          epochs=100,
          validation_data=(val_xs, val_ys),
          verbose=1,
          callbacks=[tb])

model.save('{}/model.h5'.format(DIR))
open('{}/meta_data.json'.format(DIR), 'x')
with open('{}/meta_data.json'.format(DIR), 'w') as file:
    file.write(encoder.encode({'word_index': tokenizer.word_index, 'input_len': INPUT_LEN}))
    file.close()
