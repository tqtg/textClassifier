import numpy as np
import pandas as pd
import os, sys

os.environ['KERAS_BACKEND']='tensorflow'
os.environ['CUDA_VISIBLE_DEVICES']=sys.argv[1]

from keras.utils.np_utils import to_categorical
from keras.layers import Embedding, Dropout, Reshape, GRU, Bidirectional, TimeDistributed, Dot, Activation, Dense, Input
from keras.models import Model

import preprocessor
import nltk
import itertools

# Hyper parameters
MAX_SENT_LENGTH = 300
MAX_NUM_SENTS = 20
VOCABULARY_SIZE = 50000
EMBEDDING_DIM = 200
VALIDATION_RATIO = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 20


# Data pre-processing
word_to_index = {}
def build_vocab(texts):
    global word_to_index

    # Tokenize the data into sentences
    sentences = itertools.chain(*[text.split('<sssss>') for text in texts])
    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))
    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(VOCABULARY_SIZE)
    # word index starts from 1, 0-index is reserved for padding
    word_to_index = dict([(w[0], i+1) for i, w in enumerate(vocab)])

    print("Using vocabulary size %d." % VOCABULARY_SIZE)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))


def texts_to_tensor(texts):
    global word_to_index
    documents = np.zeros((len(texts), MAX_NUM_SENTS, MAX_SENT_LENGTH), dtype='int32')
    reviews = [text.split('<sssss>') for text in texts]
    for i, review in enumerate(reviews):
        for j, sent in enumerate(review):
            if (j >= MAX_NUM_SENTS): continue
            review[j] = [w for w in sent if w in word_to_index]
            for k in range(MAX_SENT_LENGTH):
                if k < len(review[j]):
                    documents[i,j,k] = word_to_index[review[j][k]]
    return documents


# Load data (Yelp)
label_map = {1:0, 2:1, 3:2, 4:3, 5:4}

data_train = pd.read_csv('data/emnlp-2015/yelp-2013-train.txt.ss', sep='\t', header=None, usecols=[4, 6])
print(data_train.shape)
build_vocab(data_train[6])
x_train = texts_to_tensor(data_train[6])
y_train = to_categorical(data_train[4].map(label_map))

data_val = pd.read_csv('data/emnlp-2015/yelp-2013-test.txt.ss', sep='\t', header=None, usecols=[4, 6])
print(data_val.shape)
x_val = texts_to_tensor(data_val[6])
y_val = to_categorical(data_val[4].map(label_map))

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

alpha_word = TimeDistributed(Dense(1, use_bias=False))(u_word)
alpha_word = Reshape((MAX_SENT_LENGTH,))(alpha_word)
alpha_word = Activation('softmax')(alpha_word)

h_word_combined = Dot(axes=[1, 1], name='h_word_combined')([h_word, alpha_word])

sent_encoder = Model(sentence_input, h_word_combined)
sent_encoder.summary()

# Sentence level
review_input = Input(shape=(MAX_NUM_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sent_encoder, name='sent_encoder')(review_input)
l_dropout2 = Dropout(dropout)(review_encoder)

h_sent = Bidirectional(GRU(hidden_dim, return_sequences=True), name='h_sent')(l_dropout2)
u_sent = TimeDistributed(Dense(2 * hidden_dim, activation='tanh'), name='u_sent')(h_sent)

alpha_sent = TimeDistributed(Dense(1, use_bias=False))(u_sent)
alpha_sent = Reshape((MAX_NUM_SENTS,))(alpha_sent)
alpha_sent = Activation('softmax')(alpha_sent)

h_sent_combined = Dot(axes=[1, 1], name='h_sent_combined')([h_sent, alpha_sent])

# Classifier layer
l_classifier = Dense(5, activation='softmax')(h_sent_combined)

# Build model
model = Model(review_input, l_classifier)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model
model.summary()
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)