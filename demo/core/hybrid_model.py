import tensorflow as tf
from tensorflow.keras import layers, models

class HybridWeatherModel:
    def __init__(self):
        pass

    @staticmethod
    def build_model(hp=None):
        inputs = layers.Input(shape=(24, 5))
        
        # Dynamic units if hyperparameter tuner is provided
        lstm1_units = 64
        lstm2_units = 32
        drop_rate = 0.3
        
        if hp is not None:
            lstm1_units = hp.Int('lstm1_units', min_value=32, max_value=128, step=32)
            lstm2_units = hp.Int('lstm2_units', min_value=16, max_value=64, step=16)
            drop_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)

        # Recurrent Layer
        x = layers.LSTM(lstm1_units, return_sequences=True, activation='tanh')(inputs)
        x = layers.Dropout(drop_rate)(x)
        x = layers.LSTM(lstm2_units, return_sequences=True, activation='tanh')(x)
        
        # Attention Mechanism
        attention_out = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        
        # Stabilization (Residual + LayerNorm)
        x = layers.Add()([x, attention_out])
        x = layers.LayerNormalization()(x)
        
        # Aggregation
        x = layers.GlobalAveragePooling1D()(x)
        
        # Output Layer
        outputs = layers.Dense(3, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
