# data_processor/__init__.py
from sklearn.preprocessing import StandardScaler
import pandas as pd

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        self.data.fillna(self.data.mean(), inplace=True)  # Imputation
        return self.data
    def standard_scaler(self):
        try: 
            scaler = StandardScaler()
            features = self.data.drop(columns=['Value X'])
            self.data[features.columns] = scaler.fit_transform(features)
        except Exception as e:
            print("Failed to scale features with standard_scaler: " + str(e))
        return self.data    