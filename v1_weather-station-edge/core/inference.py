import os
import sys
import types
if 'imp' not in sys.modules:
    sys.modules['imp'] = types.ModuleType('imp')
import tensorflow as tf
import numpy as np

class AutoPreprocessor:
    """Auto-preprocessing pipeline for Edge Inference."""
    def __init__(self, clip_outliers=True):
        self.clip_outliers = clip_outliers
        
        # Mock historical min and max for scaling
        # Assuming features: [suhu, kelembaban, tekanan, angin, hujan]
        self.min_vals = np.array([15.0, 0.0, 950.0, 0.0, 0.0], dtype=np.float32)
        self.max_vals = np.array([45.0, 100.0, 1050.0, 30.0, 100.0], dtype=np.float32)
        
    def transform(self, raw_data):
        """Standardize and preprocess the input tensor."""
        data = np.copy(raw_data)
        
        # Safe outlier clipping
        if self.clip_outliers:
            data = np.clip(data, 0.0, 1.0)
            
        # Simulated signal smoothing (Moving Average)
        smoothed = tf.keras.layers.AveragePooling1D(pool_size=1, strides=1, padding='same')(data).numpy()
        return smoothed

class WeatherInference:
    def __init__(self, model_path="models/hybrid_model.keras", lstm_units=(64, 32), dropout_rate=0.3):
        self.model_path = model_path
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.preprocessor = AutoPreprocessor()
        self.model = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"[INFO] Model not found at {self.model_path}. Generating dummy model...")
            self.generate_dummy_model()
        else:
            print(f"[INFO] Loading model from {self.model_path}...")
            
        self.model = tf.keras.models.load_model(self.model_path)
        print("[INFO] Model loaded successfully.")

    def generate_dummy_model(self):
        """Builds a dummy model mimicking the true Hybrid LSTM-Transformer architecture."""
        inputs = tf.keras.Input(shape=(288, 5))
        
        # 1. Feature Extraction (AveragePooling)
        x = tf.keras.layers.AveragePooling1D(pool_size=12)(inputs) # shape: (24, 5)
        
        # 2. Sequential Modeling (Stacked LSTM)
        x = tf.keras.layers.LSTM(self.lstm_units[0], return_sequences=True, activation='tanh')(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.LSTM(self.lstm_units[1], return_sequences=True, activation='tanh')(x)
        
        # 3. Transformer Encoder Block
        # a. MultiHead Attention
        attention_out = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        # Residual connection + LayerNorm 1
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_out + x)
        
        # b. Feed Forward Network (FFN)
        ffn_out = tf.keras.layers.Dense(64, activation='relu')(x1)
        ffn_out = tf.keras.layers.Dense(self.lstm_units[1])(ffn_out) # match dimension
        # Residual connection + LayerNorm 2
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(ffn_out + x1)
        
        # 4. Output Block
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # 5. Model Compilation with ADAM
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model.save(self.model_path)
        print(f"[INFO] Dummy model saved to {self.model_path}")

    def predict(self, raw_data):
        """Predicts weather classification from raw input data via auto-preprocessing pipeline."""
        # Step 1: Auto-Preprocessing
        processed_data = self.preprocessor.transform(raw_data)
        
        # Step 2: Inference
        prediction = self.model.predict(processed_data, verbose=0)
        
        # Assuming classes: 0 (Cerah), 1 (Panas), 2 (Hujan)
        return int(tf.argmax(prediction, axis=1).numpy()[0])
