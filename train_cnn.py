import os

from gensim.models.keyedvectors import KeyedVectors
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from utils import cluster_quality
EMBEDDING_FILE = 'data/GoogleNews-vectors-negative300.bin'



text_path = 'data/StackOverflow.txt'
label_path = 'data/StackOverflow_gnd.txt'

with open(text_path) as f:
    data = [text.strip() for text in f]

with open(label_path) as f:
    target = f.readlines()
target = [int(label.rstrip('\n')) for label in target]

print("Total: %s short texts" % format(len(data), ","))

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

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


############################
# Preparing embedding matrix
############################


print('Preparing embedding matrix')
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

EMBEDDING_DIM = 300
nb_words = min(MAX_NB_WORDS, len(word_index)) + 1
embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
    else:
        print(word)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

#################################################
# Preparing target using Average embeddings (AE)
#################################################
Y = {}
tfidf = tokenizer.sequences_to_matrix(sequences_full, mode='tfidf')
denom = 1 + np.sum(tfidf, axis=1)[:, None]
normed_tfidf = tfidf/denom
average_embeddings = np.dot(normed_tfidf, embedding_matrix)
Y["ae"] = average_embeddings
print("Shape of average embedding: ", Y['ae'].shape)

# binary Y
from utils import binarize
reduction_name = "ae"
B = binarize(Y[reduction_name])

# Last dimension in the CNN
TARGET_DIM = B.shape[1]

# Example of binarized target vector
print(B.shape)
print(B[0])


################################################
# train model
################################################

from keras.layers import Input, Embedding, Flatten, Reshape
from keras.layers import Dense, Conv1D, Dropout, merge
from keras.layers import MaxPooling1D, GlobalMaxPooling1D, ZeroPadding1D
from keras.models import Model


def get_model():
    embedding_matrix_copy = embedding_matrix.copy()
    trainable_embedding = False
    # Embedding layer
    pretrained_embedding_layer = Embedding(
        input_dim=nb_words,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
    )

    # Input
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = pretrained_embedding_layer(sequence_input)
    
    # 1st Layer
    x = Conv1D(100, 5, activation='tanh', padding='same')(embedded_sequences)
    x = GlobalMaxPooling1D()(x)

    # Output
    x = Dropout(0.5)(x)
    predictions = Dense(TARGET_DIM, activation='sigmoid')(x)
    model = Model(sequence_input, predictions)

    model.layers[1].trainable=trainable_embedding

    adam = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # Loss and Optimizer
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['mae'])
    # Fine-tune embeddings or not
    model.summary()
    return model

if __name__ == '__main__':
    nb_epoch = 50
    checkpoint = ModelCheckpoint('models/weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    model = get_model()
    model.fit(X, B, validation_split=0.2,
              epochs=nb_epoch, batch_size=100, verbose=1, shuffle=True)

    # create model that gives penultimate layer
    input = model.layers[0].input
    output = model.layers[-2].output
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
    V = normalize(H, norm='l2')
    km.fit(V)
    pred = km.labels_
    print(pred)
    a = {'deep': cluster_quality(true_labels, pred)}
    np.save("pred.npy", pred)
    model.save_weights("model.plk")
