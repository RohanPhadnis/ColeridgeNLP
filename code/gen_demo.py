import os
import json
import numpy
from tensorflow import keras

folder = os.listdir('../models')
folder = sorted([f for f in folder if 'model' in f])
DIR = '../models/{}'.format(folder[-1])
print(DIR)

decoder = json.JSONDecoder()
with open('{}/meta_data.json'.format(DIR)) as file:
    meta_data = decoder.decode(file.read())
word_index = meta_data['word_index']
reversed_word_index = {val: key for key, val in word_index.items()}
input_len = meta_data['input_len']

model = keras.models.load_model('{}/model.h5'.format(DIR))

for _ in range(5):
    text = input('Enter starter text: ')
    text = keras.preprocessing.text.text_to_word_sequence(text)
    seq = []
    for word in text:
        if word in word_index.keys():
            seq.append(word_index[word])
        else:
            seq.append(1)
    while len(seq) < input_len:
        seq.insert(0, 0)
    for _ in range(input_len - len(text)):
        pred = list(model.predict(numpy.array([seq]))[0])
        index = pred.index(max(pred))
        text.append(reversed_word_index[index])
        seq.append(index)
        seq.pop(0)
    print(' '.join(text))
    print()
