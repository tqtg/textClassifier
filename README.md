# textClassifier

hatt_classifier.py has the implementation of [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf). See [Keras Google group discussion](https://groups.google.com/forum/#!topic/keras-users/IWK9opMFavQ) for more details.

cnn_classifier.py has implemented [Convolutional Neural Networks for Sentence Classification - Yoo Kim](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf). Some modifications in filter, pooling sizes and dropout are made.

rnn_classifier.py has implemented bidirectional LSTM and one level attentional RNN. There are two ways to incorporate attention mechanism into model by including Attention Layer or directly stacking up some default Keras core layers.
