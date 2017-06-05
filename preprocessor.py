import numpy as np
import re

from bs4 import BeautifulSoup

def clean(texts):
    processed_texts = []
    for text in texts:
        text = BeautifulSoup(text, "html.parser")
        text = text.get_text().encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r"\\", "", text)
        text = re.sub(r"\'", "", text)
        text = re.sub(r"\"", "", text)
        text.strip().lower()
        processed_texts.append(text)
    return processed_texts

def shuffle(sequences, labels):
    indices = np.arange(sequences.shape[0])
    np.random.shuffle(indices)
    sequences = sequences[indices]
    labels = labels[indices]
    return sequences, labels

def train_val_split(sequences, labels, ratio):
    sequences, labels = shuffle(sequences, labels)
    nb_validation_samples = int(ratio * sequences.shape[0])
    x_train = sequences[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = sequences[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    return x_train, y_train, x_val, y_val

def load_embedding(embedding_dim):
    embeddings_weights = {}
    f = open('embedding/glove/glove.6B.{}d.txt'.format(embedding_dim), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_weights[word] = coefs
    f.close()
    print('Total {} word vectors in Glove 6B {}d.'.format(len(embeddings_weights), embedding_dim))
    return embeddings_weights