
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "catboost_hybrid.cbm"
OUTPUT_PATH = ROOT / "model" / "catboost_feature_importance.csv"

model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))

feature_names = model.feature_names_
importance = model.get_feature_importance()

result = pd.DataFrame({
    "feature": feature_names,
    "importance": importance,
})

result = result.sort_values("importance", ascending=False)

print("\nCATBOOST FEATURE IMPORTANCE")
print(result.to_string(index=False))

result.to_csv(OUTPUT_PATH, index=False)
print("\nSaved to:", OUTPUT_PATH)
print("\nReminder: importance indicates model reliance,")
print("not proof that a feature causes fake reviews.")
