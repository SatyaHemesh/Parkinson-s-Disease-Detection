import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "parkinsons.data")

# Absolute Paths for Models
RF_MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.joblib")
SVM_MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model.joblib")
XGB_MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_model.joblib")
SCALER_SAVE_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")

TARGET = 'status'
RANDOM_STATE = 42

MODEL_PARAMS = {
    'n_estimators': 100,      
    'criterion': 'entropy',   
    'max_depth': 15,          
    'random_state': RANDOM_STATE
}

# The 4 strictly selected acoustic biomarkers
SELECTED_FEATURES = [
    'MDVP:Fo(Hz)',    
    'MDVP:Jitter(%)', 
    'MDVP:Shimmer',   
    'HNR'
]