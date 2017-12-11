from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from short_text_cnn_cluster.train_cnn import get_model


text_path = 'data/StackOverflow.txt'
label_path = 'data/StackOverflow_gnd.txt'

with open(text_path, encoding="utf-8") as f:
    data = [text.strip() for text in f]

with open(label_path, encoding="utf-8") as f:
    target = f.readlines()
target = [int(label.rstrip('\n')) for label in target]

print("Total: %s short texts" % format(len(data), ","))


tokenizer = Tokenizer(char_level=False)
tokenizer.fit_on_texts(data)
sequences_full = tokenizer.texts_to_sequences(data)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
MAX_NB_WORDS = len(word_index)

seq_lens = [len(s) for s in sequences_full]
print("Average length: %d" % np.mean(seq_lens))
print("Max length: %d" % max(seq_lens))
MAX_SEQUENCE_LENGTH = max(seq_lens)

X = pad_sequences(sequences_full, maxlen=MAX_SEQUENCE_LENGTH)
y = target


###########################################
# load cnn model
###########################################
orgin_model = get_model()

print("load weight from checkpoint...")
orgin_model.load_weights('./models/weights.010-0.7705.hdf5')

# create model that gives penultimate layer
input = orgin_model.layers[0].input
output = orgin_model.layers[-2].output
model_penultimate = Model(input, output)

# inference of penultimate layer
H = model_penultimate.predict(X)
print("Sample shape: {}".format(H.shape))


from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

true_labels = y
n_clusters = len(np.unique(y))
print("Number of classes: %d" % n_clusters)
km = KMeans(n_clusters=n_clusters, n_jobs=10)
result = dict()
pred = dict()
V = normalize(H, norm='l2')
km.fit(V)
pred['deep'] = km.labels_

print(pred)
