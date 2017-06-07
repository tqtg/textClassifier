import numpy as np
import pandas as pd
import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Flatten, Permute, Reshape, Dot, Activation

import preprocessor
import nltk
import itertools

MAX_SEQ_LENGTH = 1000
VOCABULARY_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_RATIO = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 10


# Data pre-processing
word_to_index = {}
def build_vocab(texts):
    global word_to_index
    # Tokenize the texts into words
    words = [nltk.word_tokenize(text) for text in texts]
    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*words))
    print("Found %d unique words tokens." % len(word_freq.items()))
    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(VOCABULARY_SIZE)
    # word index starts from 1, 0-index is reserved for padding
    word_to_index = dict([(w[0], i+1) for i, w in enumerate(vocab)])
    print("Using vocabulary size %d." % VOCABULARY_SIZE)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))


def texts_to_matrix(texts):
    global word_to_index
    documents = np.zeros((len(texts), MAX_SEQ_LENGTH), dtype='int32')
    reviews = [nltk.word_tokenize(text) for text in texts]
    for i, review in enumerate(reviews):
        reviews[i] = [w for w in review if w in word_to_index]
        for j in range(MAX_SEQ_LENGTH):
            if j < len(reviews[i]):
                documents[i,j] = word_to_index[reviews[i][j]]
    return documents

# Load data
data_train = pd.read_csv('data/imdb/labeledTrainData.tsv', sep='\t')
print(data_train.shape)

texts = preprocessor.clean(data_train.review)
build_vocab(texts)

documents = texts_to_matrix(texts)
labels = to_categorical(data_train.sentiment)

x_train, y_train, x_val, y_val = preprocessor.train_val_split(documents, labels, VALIDATION_RATIO)

print('Number of reviews per class in training and validation set ')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

embedding_weights = preprocessor.load_embedding(EMBEDDING_DIM)
embedding_matrix = np.zeros((VOCABULARY_SIZE + 1, EMBEDDING_DIM))
for word, i in word_to_index.items():
    embedding_vector = embedding_weights.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(VOCABULARY_SIZE + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQ_LENGTH,
                            trainable=True)

# Hyper parameters
hidden_dim = 50
dropout = 0.5

sequence_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
embedded_sequences = (embedding_layer(sequence_input))
l_dropout1 = Dropout(dropout)(embedded_sequences)

# ================================ Bidirectional LSTM model ================================

# l_lstm = Bidirectional(LSTM(hidden_dim))(l_dropout1)
# l_dropout2 = Dropout(dropout)(l_lstm)
# l_classifier = Dense(2, activation='softmax')(l_dropout2)
# model = Model(sequence_input, l_classifier)
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
# # Train model
# model.summary()
# model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)


# ================================ One-level attention RNN (GRU) ================================

h_word = Bidirectional(GRU(hidden_dim, return_sequences=True), name='h_word')(l_dropout1)

# from attention import Attention
# h_word_combined = Attention(2 * hidden_dim, name='attention')(h_word)

# Attention part
u_word = TimeDistributed(Dense(2 * hidden_dim, activation='tanh'), name='u_word')(h_word)
# \alpha weight for each word
alpha_word = TimeDistributed(Dense(1, use_bias=False))(u_word)
alpha_word = Reshape((MAX_SEQ_LENGTH,))(alpha_word)
alpha_word = Activation('softmax')(alpha_word)
# Combine word representation to form sentence representation w.r.t \alpha weights
h_word_combined = Dot(axes=[1, 1])([h_word, alpha_word])

l_classifier = Dense(2, activation='softmax', name='classifier')(h_word_combined)

model = Model(sequence_input, l_classifier)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

# ================================ Kaggle imdb submission ================================
data_test = pd.read_csv('data/imdb/testData.tsv', sep='\t')
x_test = texts_to_matrix(preprocessor.clean(data_test.review))
labels = model.predict(x_test)

submission = open('data/imdb/submission.csv', 'w')
submission.write('"id","sentiment"\n')
for i in range(len(x_test)):
    _id = data_test.id[i]
    label = np.argmax(labels[i])
    submission.write('"%s",%d\n' % (_id, label))
