import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# --- Custom Layer: Time2Vector (Positional Encoding) ---
class Time2Vector(layers.Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__()
        self.seq_len = seq_len

    def build(self, input_shape):
        # Weights untuk pola linear (tren) dan periodik (musiman)
        self.weights_linear = self.add_weight(name='weight_linear',
            shape=(int(self.seq_len), int(input_shape[-1])),
            initializer='uniform', trainable=True)
        self.bias_linear = self.add_weight(name='bias_linear',
            shape=(int(self.seq_len), int(input_shape[-1])),
            initializer='uniform', trainable=True)
        self.weights_periodic = self.add_weight(name='weight_periodic',
            shape=(int(self.seq_len), int(input_shape[-1])),
            initializer='uniform', trainable=True)
        self.bias_periodic = self.add_weight(name='bias_periodic',
            shape=(int(self.seq_len), int(input_shape[-1])),
            initializer='uniform', trainable=True)

    def call(self, x):
        # x shape: (batch, seq_len, features)
        # Kita rata-ratakan fitur untuk mendapatkan konteks waktu
        x = tf.math.reduce_mean(x, axis=-1, keepdims=True) 
        time_linear = self.weights_linear * x + self.bias_linear
        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        return tf.concat([time_linear, time_periodic], axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len})
        return config

# --- Transformer Encoder Block ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention Mechanism
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

# --- Main Model Builder ---
def create_hybrid_model(input_shape, output_shape, seq_len):
    inputs = layers.Input(shape=input_shape)
    
    # 1. Positional Encoding
    time_embedding = Time2Vector(seq_len)(inputs)
    # Gabungkan input asli dengan embedding waktu
    x = layers.Concatenate(axis=-1)([inputs, time_embedding])
    
    # 2. LSTM (Menangkap Sequential Dependencies)
    x = layers.LSTM(64, return_sequences=True, activation='tanh')(x)
    x = layers.Dropout(0.2)(x)
    
    # 3. Transformer (Menangkap Global Dependencies)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=64, dropout=0.1)
    
    # 4. Output Head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(output_shape)(x) # Output: 5 Parameter

    model = models.Model(inputs=inputs, outputs=outputs)
    return model