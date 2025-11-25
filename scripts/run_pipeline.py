# scripts/run_pipeline.py

from src.config import DATA_RAW, DATA_PICKLE, MODELS_DIR
from src.chembl_fetch import fetch_chembl_data
from src.dataset import build_feature_matrix
from src.train import train_rf
import os

# 1. Fetch data
csv_path = os.path.join(DATA_RAW, "chembl_egfr.csv")
fetch_chembl_data("CHEMBL203", csv_path)

# 2. Build features
pickle_path = os.path.join(DATA_PICKLE, "egfr_features.pkl")
X, y = build_feature_matrix(csv_path, pickle_path)

# 3. Train Random Forest
os.makedirs(MODELS_DIR, exist_ok=True)
model_path = os.path.join(MODELS_DIR, "rf_model.pkl")
model = train_rf(X, y, model_path)
