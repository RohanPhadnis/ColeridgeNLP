'''
Semantic
Analysis
Machine
Understanding
English
Literature
'''

import time
import random
import json
import numpy
from tensorflow import keras

RATIO = 0.75
VOCAB_SIZE = 5000
encoder = json.JSONEncoder()

with open('rime.txt') as file:
    text = file.read()

sentences = text.split('\n')
tokenizer = keras.preprocessing.text.Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
xs, ys = [], []
for s in sequences:
    if len(s) > 0:
        for i in range(1, len(s)):
            xs.append(s[:i])
            ys.append(s[i])
INPUT_LEN = max([len(s) for s in sequences])
xs = keras.preprocessing.sequence.pad_sequences(xs, maxlen=INPUT_LEN)
ys = keras.utils.to_categorical(ys, num_classes=VOCAB_SIZE)
print(ys)
master = [[xs[i], ys[i]] for i in range(len(xs))]
random.shuffle(master)
train, val = master[:int(len(master) * RATIO)], master[int(len(master) * RATIO):]
train_xs = []
train_ys = []
val_xs = []
val_ys = []
for i in range(len(train)):
    train_xs.append(train[i][0])
    train_ys.append(train[i][1])
for i in range(len(val)):
    val_xs.append(val[i][0])
    val_ys.append(val[i][1])

train_xs = numpy.array(train_xs)
train_ys = numpy.array(train_ys)
val_xs = numpy.array(val_xs)
val_ys = numpy.array(val_ys)

model = keras.models.Sequential([
    keras.layers.Embedding(input_dim=VOCAB_SIZE, input_length=INPUT_LEN, output_dim=16),
    keras.layers.Bidirectional(layer=keras.layers.LSTM(units=32, return_sequences=True)),
    keras.layers.Bidirectional(layer=keras.layers.LSTM(units=16)),
    keras.layers.Dense(units=32, activation='relu'),
    keras.layers.Dense(units=16, activation='relu'),
    keras.layers.Dense(units=VOCAB_SIZE, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.summary()
model.fit(x=train_xs,
          y=train_ys,
          epochs=400,
          validation_data=(val_xs, val_ys),
          verbose=1)

model.save('model{}.h5'.format(time.time()))
with open('meta.json', 'w') as file:
    file.write(encoder.encode({'word_index': tokenizer.word_index, 'input_len': INPUT_LEN}))
    file.close()
