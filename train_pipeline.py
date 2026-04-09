import os
import pandas as pd
from src.preprocessing import DataHandler
from src.model_engine import ParkinsonPredictor
import src.config as config

def run_training_pipeline():
    print("🚀 Starting Parkinson's Detection System Training Pipeline...")

    handler = DataHandler(config.DATA_PATH)
    X, y = handler.preprocess()
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test, scaler = handler.split_and_scale(X, y)
    
    predictor = ParkinsonPredictor()
    predictor.train(X_train, y_train)
    
    acc, report, cm = predictor.evaluate(X_test, y_test)
    
    print(f"\n📊 System Accuracy: {acc*100:.2f}%")
    
    # Safe key extraction regardless of pandas datatype
    target_key = '1' if '1' in report else (1 if 1 in report else list(report.keys())[0])
    try:
        print(f"✅ Precision (PD): {report[target_key]['precision']:.2f}")
        print(f"✅ Recall (PD): {report[target_key]['recall']:.2f}")
    except KeyError:
        print("⚠️ Could not extract granular precision/recall, but accuracy is calculated.")
    
    predictor.save_artifacts(scaler)
    
    # Ensure directory exists for CSV
    os.makedirs('models', exist_ok=True)
    pd.DataFrame(feature_names, columns=['feature']).to_csv('models/feature_names.csv', index=False)
    print("✅ Feature names saved.")

if __name__ == "__main__":
    run_training_pipeline()