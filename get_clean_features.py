
import joblib
import sys

try:
    with open('clean_features.txt', 'w', encoding='utf-8') as f:
        model = joblib.load('models/final_hybrid_model.pkl')
        if hasattr(model, "feature_names_in_"):
            for name in model.feature_names_in_:
                f.write(f"{name}\n")
        else:
            f.write("ERROR: No feature_names_in_")
    print("Features written to clean_features.txt")
except Exception as e:
    print(f"Error: {e}")
