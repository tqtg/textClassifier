from keras.engine.topology import Layer
from keras import backend as K

class Attention(Layer):
    '''Attention operation for temporal data.
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    '''

    def __init__(self, attention_dim, **kwargs):
        self.attention_dim = attention_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.hidden_dim = input_shape[2]
        self.W = self.add_weight(shape=(self.hidden_dim, self.attention_dim),
                                 name='att_W',
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.attention_dim,),
                                 name='att_b',
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(self.attention_dim,),
                                 name='att_u',
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.hidden_dim)

    def call(self, x, mask=None):
        # Calculate the first hidden activations
        a = K.tanh(K.dot(x, self.W) + self.b)  # [n_samples, n_steps, attention_dim]

        # K.dot won't let us dot a 3D with a 1D so we do it with mult + sum
        mul_a_u = a * self.u  # [n_samples, n_steps, attention_dim]
        dot_a_u = K.sum(mul_a_u, axis=2)  # [n_samples, n_steps]

        # Calculate the per step attention weights
        alpha_num = K.exp(dot_a_u)  # [n_samples, n_steps]
        alpha_den = K.sum(alpha_num, axis=1)  # [n_samples]
        alpha_den = K.expand_dims(alpha_den)  # [n_samples, 1] so div broadcasts
        alpha = alpha_num / alpha_den  # [n_samples, n_steps]
        alpha = K.expand_dims(alpha)  # [n_samples, n_steps, 1] so div broadcasts

        # Apply attention weights to steps
        weighted_input = x * alpha  # [n_samples, n_steps, n_features]

        # Sum across the weighted steps to get the pooled activations
        return K.sum(weighted_input, axis=1)