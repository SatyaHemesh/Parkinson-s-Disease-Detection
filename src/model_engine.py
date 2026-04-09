import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import src.config as config

class ParkinsonPredictor:
    def __init__(self):
        self.models = {
            "Random Forest": RandomForestClassifier(**config.MODEL_PARAMS),
            "SVM": SVC(probability=True, random_state=config.RANDOM_STATE),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=config.RANDOM_STATE)
        }

    def train(self, X_train, y_train):
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            print(f"✅ {name} training complete.")

    def evaluate(self, X_test, y_test):
        predictions = self.models["Random Forest"].predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        cm = confusion_matrix(y_test, predictions)
        return accuracy, report, cm

    def save_artifacts(self, scaler):
        # Prevent FileNotFoundError by creating the directory
        os.makedirs(os.path.dirname(config.RF_MODEL_PATH), exist_ok=True)
        
        joblib.dump(self.models["Random Forest"], config.RF_MODEL_PATH)
        joblib.dump(self.models["SVM"], config.SVM_MODEL_PATH)
        joblib.dump(self.models["XGBoost"], config.XGB_MODEL_PATH)
        joblib.dump(scaler, config.SCALER_SAVE_PATH)
        print("✅ All 3 models saved successfully to the models/ folder.")