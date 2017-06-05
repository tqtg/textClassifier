import numpy as np
import pandas as pd
import sys
import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.layers import Dense, Input, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Flatten, merge, Permute, Reshape

from keras import backend as K

import preprocessor

MAX_SEQ_LENGTH = 400
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_RATIO = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 10

# Load data
data_train = pd.read_csv('data/imdb/labeledTrainData.tsv', sep='\t')
print(data_train.shape)

x_train = preprocessor.clean(data_train.review)
y_train = to_categorical(data_train.sentiment)

# Tokenize data and map token to unique id
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(x_train)
x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, maxlen=MAX_SEQ_LENGTH)
word_index = tokenizer.word_index
print('Found %d unique tokens.' % len(word_index))

x_train, y_train, x_val, y_val = preprocessor.train_val_split(x_train, y_train, VALIDATION_RATIO)

print('Number of positive and negative reviews in training and validation set ')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

embedding_weights = preprocessor.load_embedding(EMBEDDING_DIM)
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embedding_weights.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQ_LENGTH,
                            trainable=True)

# Construct model
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

def get_weighted_sum(X):
    h, alpha = X[0], X[1]
    ans = K.batch_dot(h, alpha)
    return ans

h_word = Bidirectional(GRU(hidden_dim, return_sequences=True), name='Bidirect_GRU')(l_dropout1)

u_word = TimeDistributed(Dense(2 * hidden_dim, activation='tanh'), name='u_word')(h_word)
alpha_word = TimeDistributed(Dense(1, activation='linear'))(u_word)
flat_alpha = Flatten()(alpha_word)
alpha_word = Dense(MAX_SEQ_LENGTH, activation='softmax', name='alpha_word')(flat_alpha)

h_word_trans = Permute((2, 1), name="h_word_trans")(h_word)
h_word_combined = merge([h_word_trans, alpha_word], output_shape=(2 * hidden_dim, 1), name="h_word_combined", mode=get_weighted_sum)
h_word_combined = Reshape((2 * hidden_dim,), name="reshape")(h_word_combined)

l_classifier = Dense(2, activation='softmax')(h_word_combined)

model = Model(sequence_input, l_classifier)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)