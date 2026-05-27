import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, scaler_path="models/scaler.pkl"):
        self.scaler_path = scaler_path
        self.scaler = MinMaxScaler()
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)

    def handle_missing_data(self, df):
        """Impute missing data using linear interpolation up to 6 hours."""
        # Frequency is 5 minutes. 6 hours = 6 * 12 = 72 samples limit.
        df_imputed = df.interpolate(method='linear', limit=72, limit_direction='both')
        return df_imputed

    def aggregate_data(self, df):
        """Aggregate 288 samples (5 min freq) into 24 samples (hourly) via Average Pooling."""
        # Expecting df to have 288 rows. We group by every 12 samples (1 hour).
        if len(df) != 288:
            raise ValueError("Expected 288 samples for a full day.")
        
        # Reshape to (24, 12, num_features) and average over axis 1
        arr = df.to_numpy()
        arr_reshaped = arr.reshape(24, 12, -1)
        aggregated = np.mean(arr_reshaped, axis=1)
        
        # Convert back to dataframe for consistent scaling
        return pd.DataFrame(aggregated, columns=df.columns)

    def scale_data(self, data, fit=False):
        """Scale data using MinMaxScaler and save/load scaler."""
        if fit:
            scaled = self.scaler.fit_transform(data)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        else:
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            else:
                # If no scaler exists, fit it on current data as fallback
                scaled = self.scaler.fit_transform(data)
                with open(self.scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            scaled = self.scaler.transform(data)
            
        # Return reshaped for LSTM input (1, 24, 5)
        return scaled.reshape(1, 24, -1)
