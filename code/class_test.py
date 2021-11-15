import os
import json
import numpy
from tensorflow import keras

folder = os.listdir('../models')
folder = sorted([f for f in folder if 'classifier' in f])
DIR = '../models/{}'.format(folder[-1])
print(DIR)
decoder = json.JSONDecoder()

model = keras.models.load_model('{}/model.h5'.format(DIR))
test = 'The night is bright.'
test = keras.preprocessing.text.text_to_word_sequence(test)

with open('{}/meta_data.json'.format(DIR)) as file:
    meta_data = decoder.decode(file.read())

word_index = meta_data['word_index']
INPUT_LEN = meta_data['input_len']
seq = []
for word in test:
    if word in word_index.keys():
        seq.append(word_index[word])
    else:
        seq.append(1)

while len(seq) < INPUT_LEN:
    seq.insert(0, 0)

seq = numpy.array([seq])
pred = model.predict(seq)[0][0]
print(pred)

open('{}/meta.tsv'.format(DIR), 'x')
open('{}/vecs.tsv'.format(DIR), 'x')

meta = open('{}/meta.tsv'.format(DIR), 'a')
vecs = open('{}/vecs.tsv'.format(DIR), 'a')

weights = model.layers[0].weights[0]
reversed_word_index = {val: key for key, val in word_index.items()}
for i in range(5000):
    output = []
    if i in reversed_word_index.keys():
        meta.write(reversed_word_index[i] + '\n')
        for j in weights[i]:
            output.append(str(float(j)))
        vecs.write('\t'.join(output) + '\n')

meta.close()
vecs.close()
