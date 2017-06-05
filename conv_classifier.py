import numpy as np
import pandas as pd

import os, sys
os.environ['KERAS_BACKEND']='tensorflow'
os.environ['CUDA_VISIBLE_DEVICES']=sys.argv[1]

from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Convolution1D, MaxPooling1D, Embedding, Concatenate, Dropout
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import preprocessor

MAX_SEQ_LENGTH = 400
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_RATIO = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 10

# Load data
data_train = pd.read_csv('data/imdb/labeledTrainData.tsv', sep='\t')
data_test = pd.read_csv('data/imdb/testData.tsv', sep='\t')
print(data_train.shape)
print(data_test.shape)

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
conv_layers = []
filter_sizes = [3, 4, 5]
num_filters = 10
bottleneck_dim = 128
dropout = 0.5

sequence_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
embedded_sequences = Dropout(dropout)(embedding_layer(sequence_input))

for fsz in filter_sizes:
    l_conv = Convolution1D(filters=num_filters,
                           kernel_size=fsz,
                           padding='valid',
                           activation='relu')(embedded_sequences)
    l_pool = MaxPooling1D(pool_size=5)(l_conv)
    l_flat = Flatten()(l_pool)
    conv_layers.append(l_flat)

l_concat = Concatenate()(conv_layers)
l_dropout1 = Dropout(dropout)(l_concat)
l_bottleneck = Dense(bottleneck_dim, activation='relu')(l_dropout1)
l_dropout2 = Dropout(dropout)(l_bottleneck)
l_classifier = Dense(2, activation='softmax')(l_dropout2)

model = Model(sequence_input, l_classifier)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

# Test data for submission
x_test = preprocessor.clean(data_test.review)
x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, maxlen=MAX_SEQ_LENGTH)
labels = model.predict(x_test)

submission = open('data/imdb/submission.csv', 'w')
submission.write('"id","sentiment"\n')
for i in range(len(x_test)):
    _id = data_test.id[i]
    label = np.argmax(labels[i])
    submission.write('"%s",%d\n' % (_id, label))