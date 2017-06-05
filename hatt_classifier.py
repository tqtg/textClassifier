import numpy as np
import pandas as pd
import os

os.environ['KERAS_BACKEND']='tensorflow'

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input, Flatten
from keras.layers import Embedding, Dropout, Permute, Reshape, GRU, Bidirectional, TimeDistributed, Dot
from keras.models import Model

from nltk import tokenize as nltk_tokenize

import preprocessor

MAX_SENT_LENGTH = 100
MAX_NUM_SENTS = 10
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_RATIO = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 10

# Load data
data_train = pd.read_csv('data/imdb/labeledTrainData.tsv', sep='\t')
print(data_train.shape)

texts = preprocessor.clean(data_train.review)
labels = to_categorical(data_train.sentiment)

# Tokenize data and map token to unique id
reviews = []
for text in texts:
    sentences = nltk_tokenize.sent_tokenize(text)
    reviews.append(sentences)

tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer.fit_on_texts(texts)

word_index = tokenizer.word_index
print('Found %d unique tokens.' % len(word_index))

data = np.zeros((len(texts), MAX_NUM_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(reviews):
    for j, sent in enumerate(sentences):
        if j < MAX_NUM_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for _, word in enumerate(wordTokens):
                if k < MAX_SENT_LENGTH and word_index[word] < MAX_NUM_WORDS:
                    data[i,j,k] = word_index[word]
                    k += 1


x_train, y_train, x_val, y_val = preprocessor.train_val_split(data, labels, VALIDATION_RATIO)

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
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)

# Hyper parameters
hidden_dim = 50
dropout = 0.5

# Word level
sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = (embedding_layer(sentence_input))
l_dropout1 = Dropout(dropout)(embedded_sequences)

h_word = Bidirectional(GRU(hidden_dim, return_sequences=True), name='h_word')(l_dropout1)
u_word = TimeDistributed(Dense(2 * hidden_dim, activation='tanh'), name='u_word')(h_word)

alpha_word = TimeDistributed(Dense(1, activation='linear'))(u_word)
flat_alpha_word = Flatten()(alpha_word)
alpha_word = Dense(MAX_SENT_LENGTH, activation='softmax', name='alpha_word')(flat_alpha_word)

h_word_trans = Permute((2, 1), name='h_word_trans')(h_word)
h_word_combined = Dot(axes=[2, 1], name='h_word_combined')([h_word_trans, alpha_word])
h_word_combined = Reshape((2 * hidden_dim,), name="reshape")(h_word_combined)

sent_encoder = Model(sentence_input, h_word_combined)

# Sentence level
review_input = Input(shape=(MAX_NUM_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sent_encoder, name='sent_encoder')(review_input)
l_dropout2 = Dropout(dropout)(review_encoder)

h_sent = Bidirectional(GRU(hidden_dim, return_sequences=True), name='h_sent')(l_dropout2)
u_sent = TimeDistributed(Dense(2 * hidden_dim, activation='tanh'), name='u_sent')(h_sent)

alpha_sent = TimeDistributed(Dense(1, activation='linear'))(u_sent)
flat_alpha_sent = Flatten()(alpha_sent)
alpha_sent = Dense(MAX_NUM_SENTS, activation='softmax', name='alpha_sent')(flat_alpha_sent)

h_sent_trans = Permute((2, 1), name='h_sent_trans')(h_sent)
h_sent_combined = Dot(axes=[2, 1], name='h_sent_combined')([h_sent_trans, alpha_sent])
h_sent_combined = Reshape((2 * hidden_dim,), name="reshape")(h_sent_combined)

# Classifier layer
l_classifier = Dense(2, activation='softmax')(h_sent_combined)

model = Model(review_input, l_classifier)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)