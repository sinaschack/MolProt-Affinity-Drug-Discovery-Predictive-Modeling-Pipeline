# MolProt-Affinity-Drug-Discovery-Predictive-Modeling-Pipeline 

End-to-end predictive modeling workflow in Python for small molecule–protein binding affinity. Pulls real ChEMBL bioactivity data, featurizes molecules and proteins, trains predictive models (Random Forest, XGBoost, and Deep Learning), and offers basic interpretability with inline plots.

--- 

## Overview 

The project guides through an end-to-end workflow for small molecule binding to a protein target. EGFR is used as an example target throughout the project.

The workflow includes: 

* Fetching ChEMBL IC50 bioactivity data for a target protein
* Data cleaning and exploration 
* Molecular featurization: Morgan fingerprints and RDKit descriptors 
* Protein featurization: amino acid composition and 2-mer counts
* Creating a final feature matrix combining molecule and protein features
* Training baseline machine learning models (Random Forest and XGBoost)
* Training PyTorch feed-forward neural networks (FFNN) 
* Inline visualization of pIC50 distributions and potency-colored molecule grids

--- 

## Installation 

1. Clone the repository and enter the directory: 

```bash 
git clone <repo-url> 
cd MolProt-Affinity 
``` 

2. Create a virtual environment and activate it: 

```bash 
python -m venv venv 
source venv/bin/activate # Linux/macOS 
venv\Scripts\activate # Windows 
``` 

3. Install the dependencies: 

```bash 
pip install -r requirements.txt 
``` 

--- 

## Usage 

### 1. Fetch ChEMBL bioactivity data 

Fetch IC50 data for a given target (CHEMBL ID) and save it as CSV:

```python 
from src.chembl_fetch import fetch_chembl_data 

df = fetch_chembl_data("CHEMBL203", "data/raw/chembl_egfr.csv") 
``` 

This function will return a DataFrame containing: 

* `smiles` — canonical SMILES string for the molecule 
* `IC50_nM` — experimental IC50 
* `pIC50` — converted target value (`pIC50 = 9 - log10(IC50_nM)`) 

--- 

### 2. Data cleaning & exploration 

Drop duplicates and visualize pIC50 distribution. 

```python 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

df = pd.read_csv("data/raw/chembl_egfr.csv") 
df = df.drop_duplicates(subset=["smiles"]) 

# Histogram 
sns.histplot(df["pIC50"], bins=30, kde=True) 
plt.show() 

# Boxplot 
sns.boxplot(x=df["pIC50"]) 
plt.show() 
``` 

Create a potency-colored molecule grid using RDKit. 

--- 

### 3. Feature Engineering 

**Molecular features:** 

* Morgan fingerprints (1024-bit) 
* RDKit descriptors: MolWt, LogP, TPSA, HDonors, HAcceptors, RotBonds

**Protein features:** 

* Amino acid composition (20-dimensional) 
* 2-mer composition (400-dimensional) 

**Final feature matrix:** 

`X_final` = [Morgan + RDKit + Protein] → `(N_molecules, 1450 features)` 

It's a good idea to pickle preprocessed features for reproducibility.

```python 
import pickle 
with open("data/pickle/egfr_combined.pkl", "wb") as f: 
pickle.dump({ 
"X_combined": X_final, 
"y": y, 
"smiles": df["smiles"].tolist() 
}, f) 
``` 

--- 

### 4. Baseline Machine Learning Models 

Train Random Forest and XGBoost regressors on the combined feature matrix.

```python 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
import xgboost as xgb 
import joblib 

# Split 
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Random Forest 
rf = RandomForestRegressor(n_estimators=600, n_jobs=-1, random_state=42) 
rf.fit(X_train, y_train) 
joblib.dump(rf, "models/random_forest.pkl") 

# XGBoost 
xgb_model = xgb.XGBRegressor( 
n_estimators=1000, learning_rate=0.05, max_depth=8, 
subsample=0.8, colsample_bytree=0.8, random_state=42 
) 
xgb_model.fit(X_train, y_train) 
joblib.dump(xgb_model, "models/xgboost.pkl") 
``` 

Evaluation (RMSE, R², etc.) can be done on `X_test` / `y_test`. 

--- 

### 5. Deep Learning Models (PyTorch) 

Three feed-forward neural networks are implemented: 

* `FFNN_Simple` – 2 hidden layers, ReLU + dropout 
* `FFNN_Deep` – deeper network with BatchNorm, dropout 
* `FFNN_Wide` – wider network with more neurons per layer

**Training:** 

```python 
from src.deep_learning import FFNN_Simple, FFNN_Deep, FFNN_Wide, train_model 

train_model(FFNN_Simple(input_dim=X_final.shape[1]), "ffnn_egfr_simple") 
train_model(FFNN_Deep(input_dim=X_final.shape[1]), "ffnn_egfr_deep") 
train_model(FFNN_Wide(input_dim=X_final.shape[1]), "ffnn_egfr_wide") 
``` 

Models will be automatically saved as `.pth` files in `models/` directory. Early stopping is used to avoid overfitting. 

--- 

## Visualization 

* Histograms and boxplots of pIC50 values 
* Potency-colored molecule grids (green = strong binder, red = weak binder)

--- 

## Data & Features 

| Feature Type | Dimensions | Description | 
| ---------------------- | ---------- | ------------------------------------------------ | 
| Morgan fingerprints | 1024 | Circular fingerprints of molecules | 
| RDKit descriptors | 6 | MolWt, LogP, TPSA, HDonors, HAcceptors, RotBonds |
| Protein AA composition | 20 | Fraction of each amino acid |
| Protein 2-mer | 400 | 2-mer counts normalized by sequence length |

**Total features:** 1450 per molecule–protein pair. 

--- 

## References 

* [ChEMBL Web Resource Client](https://github.com/chembl/chembl_webresource_client) 
* [RDKit](https://www.rdkit.org/) 
* [Scikit-learn](https://scikit-learn.org/) 
* [XGBoost](https://xgboost.readthedocs.io/) 
* [PyTorch](https://pytorch.org/) 
