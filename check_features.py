import joblib
import json

# Load model info
with open('feature_names.json', 'r') as f:
    feature_info = json.load(f)

print("="*60)
print("MODEL FEATURE REQUIREMENTS")
print("="*60)
print(f"Total features expected: {len(feature_info['feature_names'])}")
print("\nTop 20 features:")
for i, feat in enumerate(feature_info['feature_names'][:20]):
    print(f"{i+1:2}. {feat}")

print(f"\nModel Performance:")
print(f"R² Score: {feature_info['performance']['r2_score']}")
print(f"MAPE: {feature_info['performance']['mape']}%")