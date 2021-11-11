import json
import numpy
from tensorflow import keras

decoder = json.JSONDecoder()
with open('../model/meta.json') as file:
    meta_data = decoder.decode(file.read())
model = keras.models.load_model('../model/model1636597371.8986976.h5')
text = 'it is an'
text = keras.preprocessing.text.text_to_word_sequence(text)
seq = []
for i in range(len(text)):
    if text[i] in meta_data['word_index'].keys():
        seq.append(meta_data['word_index'][text[i]])
    else:
        seq.append(1)
while len(seq) < meta_data['input_len']:
    seq.insert(0, 0)
reversed_word_index = {val: key for key, val in meta_data['word_index'].items()}
for i in range(meta_data['input_len'] - len(text)):
    pred = list(model.predict(numpy.array([seq]))[0])
    print(len(pred))
    seq.append(pred.index(max(pred)))
    seq.pop(0)
text = [reversed_word_index[n] for n in seq]
print(text)
meta = open('../model/meta.tsv', 'w')
meta.write('')
meta.close()
vecs = open('../model/vecs.tsv', 'w')
vecs.write('')
vecs.close()
meta = open('../model/meta.tsv', 'a')
vecs = open('../model/vecs.tsv', 'a')
for i in range(5000):
    if i in reversed_word_index.keys():
        meta.write(reversed_word_index[i] + '\n')
        output = []
        for j in range(16):
            output.append(str(float(model.layers[0].weights[0][i][j])))
        vecs.write('\t'.join(output) + '\n')
meta.close()
vecs.close()
print('done')
