"""
train.py — Standalone Training Script
======================================
Trains the Laptop Spec Tier Classifier and saves the model to disk.

Usage:
    python train.py
    python train.py --data data/laptops.csv --output model/final_model.pkl

The script will:
    1. Load and feature-engineer the dataset
    2. Generate SpecTier labels via SpecScore + qcut
    3. Train XGBoost + Random Forest Soft-Voting Ensemble
    4. Evaluate on a held-out 20% test set
    5. Save the trained model as a .pkl file
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


# ─────────────────────────────────────────────
# FEATURE ENGINEERING  (same as app.py)
# ─────────────────────────────────────────────
def parse_storage(mem_str):
    mem_str = str(mem_str).upper()
    total_gb = 0
    for v in pd.Series([mem_str]).str.findall(r'(\d+)TB')[0]:
        total_gb += int(v) * 1024
    for v in pd.Series([mem_str]).str.findall(r'(\d+)GB')[0]:
        total_gb += int(v)
    if 'NVME' in mem_str or 'PCIE' in mem_str:
        return total_gb, 'NVMe', 4
    elif 'SSD' in mem_str:
        return total_gb, 'SSD', 3
    elif 'FUSION' in mem_str:
        return total_gb, 'Fusion', 2
    elif 'HDD' in mem_str:
        return total_gb, 'HDD', 1
    elif 'SSHD' in mem_str:
        return total_gb, 'SSHD', 2
    return total_gb, 'Flash', 1


def cpu_tier(c):
    c = str(c).upper()
    if 'XEON' in c or 'I7' in c:
        return 3
    elif 'I5' in c:
        return 2
    elif 'I3' in c:
        return 1
    return 0


def gpu_tier(g):
    g = str(g).upper()
    if 'RX 5600' in g:
        return 2
    if 'GTX 1650' in g:
        return 1
    return 0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Ram_GB']    = df['Ram'].astype(str).str.extract(r'(\d+)').astype(int)
    df['Weight_kg'] = pd.to_numeric(df['Weight'], errors='coerce')
    df['Inches']    = pd.to_numeric(df['Inches'], errors='coerce')

    storage_parsed     = df['Memory'].apply(lambda x: pd.Series(parse_storage(x)))
    df['Storage_GB']   = storage_parsed[0]
    df['StorageType']  = storage_parsed[1]
    df['StorageSpeed'] = storage_parsed[2]

    df['CPU_Tier']  = df['Cpu'].apply(cpu_tier)
    df['CPU_Brand'] = df['Cpu'].apply(lambda x: 'AMD' if 'AMD' in str(x) else 'Intel')

    df['GPU_Tier']  = df['Gpu'].apply(gpu_tier)
    df['GPU_Brand'] = df['Gpu'].apply(
        lambda x: 'Nvidia' if 'NVIDIA' in str(x).upper()
        else ('AMD' if 'AMD' in str(x) else 'Intel')
    )

    sr = df['ScreenResolution'].astype(str)
    df['Is_Touchscreen'] = sr.str.contains('Touchscreen').astype(int)
    df['Is_IPS']         = sr.str.contains('IPS').astype(int)
    df['Is_4K']          = sr.str.upper().str.contains('4K|3840').astype(int)
    df['Is_2K']          = sr.str.upper().str.contains('2K|2560').astype(int)
    df['Is_Retina']      = sr.str.contains('Retina').astype(int)

    df['IsPremiumBrand']  = df['CompanyName'].isin(['Apple', 'Microsoft', 'Dell']).astype(int)
    df['IsGaming']        = df['TypeOfLaptop'].str.contains('Gaming').astype(int)
    df['IsWorkstation']   = df['TypeOfLaptop'].str.contains('WorkStation').astype(int)
    df['IsConvertible']   = df['TypeOfLaptop'].str.contains('Convertible').astype(int)
    df['IsUltrabook']     = df['TypeOfLaptop'].str.contains('UltraBook').astype(int)

    df['GPU_Tier_Score'] = df['GPU_Tier'].map({0: 1.0, 1: 1.5, 2: 2.2})
    df['PowerScore']     = (df['Ram_GB'] * (df['CPU_Tier'] + 1) * df['GPU_Tier_Score']).round(2)
    df['SpecScore']      = (
        df['Ram_GB'] * df['StorageSpeed'] *
        (df['GPU_Tier'] + 1) * (df['CPU_Tier'] + 1)
    ).astype(float)
    return df


FEATURE_COLS = [
    'Ram_GB', 'Storage_GB', 'StorageSpeed', 'CPU_Tier', 'GPU_Tier',
    'Is_Touchscreen', 'Is_IPS', 'Is_4K', 'Is_2K', 'Is_Retina',
    'Inches', 'Weight_kg',
    'IsPremiumBrand', 'IsGaming', 'IsWorkstation', 'IsConvertible', 'IsUltrabook',
    'PowerScore',
    'TypeOfLaptop', 'StorageType', 'GPU_Brand', 'CPU_Brand', 'OpSys', 'CompanyName',
]
CAT_COLS = ['TypeOfLaptop', 'StorageType', 'GPU_Brand', 'CPU_Brand', 'OpSys', 'CompanyName']


# ─────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────
def train(data_path: str, output_path: str, verbose: bool = True):
    # 1. Load
    if verbose:
        print(f"[1/5] Loading data from: {data_path}")
    df_raw = pd.read_csv(data_path)
    if verbose:
        print(f"      Shape: {df_raw.shape}")

    # 2. Feature engineering
    if verbose:
        print("[2/5] Engineering features...")
    df = engineer_features(df_raw)
    df['SpecTier'] = pd.qcut(
        df['SpecScore'], q=3,
        labels=['Basic', 'Mainstream', 'Powerhouse'],
        duplicates='drop'
    )
    if verbose:
        print("      Tier distribution:")
        print(df['SpecTier'].value_counts().to_string())

    # 3. Prepare matrices
    if verbose:
        print("[3/5] Preparing feature matrix...")
    df_model = df[FEATURE_COLS + ['SpecTier']].copy().dropna(subset=['SpecTier'])
    df_model = pd.get_dummies(df_model, columns=CAT_COLS, drop_first=True)

    le = LabelEncoder()
    y  = le.fit_transform(df_model['SpecTier'].astype(str))
    X  = df_model.drop('SpecTier', axis=1).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    if verbose:
        print(f"      Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # 4. Train
    if verbose:
        print("[4/5] Training ensemble...")

    xgb_clf = XGBClassifier(
        n_estimators=300, learning_rate=0.10, max_depth=6,
        subsample=0.85, colsample_bytree=0.80, min_child_weight=2,
        gamma=0.05, reg_alpha=0.1, reg_lambda=1.2,
        random_state=42, eval_metric='mlogloss',
    )
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=4,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_clf), ('rf', rf_clf)],
        voting='soft',
    )
    ensemble.fit(X_train, y_train)

    # 5. Evaluate
    if verbose:
        print("[5/5] Evaluating...")

    y_pred  = ensemble.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    report  = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    cm      = confusion_matrix(y_test, y_pred)

    xgb_clf.fit(X_train, y_train)
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc  = cross_val_score(xgb_clf, X_train, y_train, cv=cv, scoring='accuracy')

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Test Accuracy    : {acc*100:.2f}%")
        print(f"  CV Accuracy      : {cv_acc.mean()*100:.2f}% ± {cv_acc.std()*100:.2f}%")
        print(f"{'='*50}")
        print("\nClassification Report:")
        print(report)
        print("Confusion Matrix:")
        print(pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_string())

    # Save model + metadata
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    bundle = {
        "model":        ensemble,
        "le":           le,
        "feature_cols": list(X.columns),
    }
    with open(output_path, 'wb') as f:
        pickle.dump(bundle, f)

    if verbose:
        print(f"\n✅ Model saved to: {output_path}")

    return acc, cv_acc


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Laptop Spec Tier Classifier")
    parser.add_argument("--data",    default="data/laptops.csv",     help="Path to laptops.csv")
    parser.add_argument("--output",  default="model/final_model.pkl", help="Output path for saved model")
    parser.add_argument("--quiet",   action="store_true",             help="Suppress output")
    args = parser.parse_args()

    train(data_path=args.data, output_path=args.output, verbose=not args.quiet)
