import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Positional Encoding Function
def positional_encoding(seq_len, model_dim):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(model_dim)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(model_dim))
    angle_rads = pos * angle_rates

    # Apply sin to even indices in the array; cos to odd indices
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# Multi-head attention mechanism
class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, model_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim

        assert model_dim % num_heads == 0

        self.depth = model_dim // num_heads

        self.wq = layers.Dense(model_dim)
        self.wk = layers.Dense(model_dim)
        self.wv = layers.Dense(model_dim)

        self.dense = layers.Dense(model_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.model_dim))
        return self.dense(concat_attention)

# Scaled Dot-Product Attention
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

class TransformerBlock(layers.Layer):
    def __init__(self, model_dim, num_heads, ff_dim, trainable=True, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.trainable = trainable  # Store the trainable argument

        # Define layers
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.model_dim,
            name="multi_head_attention"
        )
        self.ffn = keras.Sequential([
            layers.Dense(self.ff_dim, activation='relu', name="dense_1"),
            layers.Dense(self.model_dim, name="dense_2")
        ], name="feed_forward_network")
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def call(self, inputs, training=None):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'model_dim': self.model_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'trainable': self.trainable  # Include trainable in the config
        })
        return config

# Build the Transformer Model
def build_transformer_model(vocab_size, max_len, model_dim, num_heads, ff_dim, num_layers):
    inputs = layers.Input(shape=(max_len,))

    # Embedding and Positional Encoding
    x = layers.Embedding(vocab_size, model_dim)(inputs)
    x += positional_encoding(max_len, model_dim)

    # Add multiple transformer blocks
    for _ in range(num_layers):
        x = TransformerBlock(model_dim, num_heads, ff_dim)(x)

    # Final Dense layer for next-token prediction
    outputs = layers.Dense(vocab_size, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
