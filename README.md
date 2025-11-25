# MolProt-Affinity-Drug-Discovery-Predictive-Modeling-Pipeline
End-to-end drug discovery machine learning workflow in Python. Retrieves real ChEMBL bioactivity data, featurizes molecules and proteins, trains predictive models (Random Forest, XGBoost), and offers interpretability with SHAP.

## Overview

This project walks through an end-to-end predictive modeling pipeline for small molecule–protein binding affinity, with EGFR used as a concrete example target. Key steps include:

* Data collection from ChEMBL in the form of canonical SMILES strings and IC50 bioactivity values
* Molecular featurization with Morgan fingerprints and RDKit descriptors
* Protein featurization with amino acid composition and k-mer counts
* Building combined molecule–protein feature matrices as NumPy arrays
* Fitting baseline ML models for regression (Random Forest, XGBoost)
* Evaluating model performance and generalization (RMSE, R², cross-validation)
* Providing interpretability with SHAP to understand model predictions
* Visualizing molecular potency grids and distribution plots

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

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1.

Fetch ChEMBL bioactivity data for a given target (Chemical Identifier + path to save CSV file).

```python
from src.chembl_fetch import fetch_chembl_data

df = fetch_chembl_data("CHEMBL203", "data/raw/chembl_egfr.csv")
```

### 2.

Generate combined molecule and protein feature matrix with labels (input CSV file path + output pickle path).

```python
from src.dataset import build_feature_matrix

X, y = build_feature_matrix("data/raw/chembl_egfr.csv", "data/pickle/egfr_combined.pkl")
```

### 3.

Train a predictive model on the preprocessed feature matrix.

```python
from src.train import train_rf
import joblib

model = train_rf(X, y, "models/random_forest.pkl")
```

Baseline models for Random Forest and XGBoost regression are implemented and can be trained in a similar manner.

## Deep Learning Models

In addition to the classical ML models, feed-forward neural networks (FFNNs) were trained using combined molecule and protein features (1450 dimensions). Three variants were implemented:

1. **FFNN Simple (`ffnn_egfr_simple.pth`)**  
   - 3 layers: 1450 → 512 → 128 → 1  
   - ReLU activations, dropout 0.2  
   - Baseline deep learning model for comparison with classical ML

2. **FFNN Deep (`ffnn_egfr_deep.pth`)**  
   - 4 layers with batch normalization and dropout  
   - Wider and deeper to capture more complex relationships

3. **FFNN Wide (`ffnn_egfr_wide.pth`)**  
   - Wider hidden layers: 1450 → 1024 → 512 → 256 → 1  
   - ReLU activations, moderate dropout

### Training and Usage

```python
from src.models.train_dl import train_model
from src.models.ffnn import FFNN_Simple, FFNN_Deep, FFNN_Wide

# Example: train the simple FFNN
model = FFNN_Simple(input_dim=1450)
train_model(model, name="ffnn_egfr_simple")

## Model Evaluation

The models were evaluated in a variety of ways, including:

* Train/test split performance (RMSE, R²)
* K-fold cross-validation with shuffling
* Train/valid/test splits with cross-validation
* Hyperparameter tuning (Randomized Search, Grid Search)

Performance was measured using RMSE loss and R² coefficient of determination. Feature importance and SHAP values were also computed for interpretability.

## Interpretability

SHAP feature importance was calculated and visualized for the trained models using fitted feature importances and SHAP summary plots.

```python
from src.explain import explain_model

explain_model(model, X_test, feature_names)
```

The importance of molecular features, protein features, and all features are visualized in HTML static reports.

## Visualization

Molecule grids colored by potency (IC50 values) and distribution plots were also generated to help understand the data.

## Data and Features

The following features are used to describe each molecule and its target protein. The raw data was sourced from the [ChEMBL database](https://www.ebi.ac.uk/chembl/) and downloaded using [chembl_webresource_client](https://github.com/chembl/chembl_webresource_client).

### Molecular features

Molecular features are extracted from the SMILES strings using RDKit:

* Morgan fingerprints (1024-bit)
* RDKit descriptors: MolWt, LogP, TPSA, HDonors, HAcceptors, RotBonds

### Protein features

Protein features are derived from the sequences of the target proteins:

* Amino acid composition (20-dimensional)
* 2-mer amino acid composition (400-dimensional)

### Combined feature matrix

The final feature matrix (X) contains both molecular and protein features concatenated for each molecule–protein pair:

* Example shape for 365 EGFR-target molecules: (365, 1450) features per sample

The labels (y) are the continuous pIC50 values converted from the IC50 values.

## Results

### Model performance

RMSE loss and R² coefficient of determination for the baseline models:

* RF baseline: RMSE ~0.72
* XGBoost baseline: RMSE ~0.83

A higher R² value close to 1 indicates that the predicted pIC50 values are strongly correlated to the experimental values.

### Visualization

Potency-colored molecule grids and distribution plots were also generated to help understand the dataset.

## References

* [ChEMBL Web Resource Client](https://github.com/chembl/chembl_webresource_client)
* [RDKit](https://www.rdkit.org/)
* [Scikit-learn](https://scikit-learn.org/)
* [XGBoost](https://xgboost.readthedocs.io/)
* [SHAP](https://github.com/slundberg/shap)
