import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import src.config as config

class DataHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None
        self.scaler = StandardScaler()

    def load_data(self):
        try:
            self.data = pd.read_csv(self.filepath)
            print(f"✅ Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            raise Exception("❌ Dataset not found. Please place 'parkinsons.data' in the data/ folder.")

    def preprocess(self):
        if self.data is None:
            self.load_data()

        # Strictly filter to the 4 selected features
        X = self.data[config.SELECTED_FEATURES]
        y = self.data[config.TARGET]
        
        return X, y

    def split_and_scale(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, self.scaler