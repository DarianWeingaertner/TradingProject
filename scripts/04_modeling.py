# scripts/04_modeling.py
"""
Step 04 — Modeling
------------------
Modelle:
- Logistic Regression (interpretable)
- Random Forest (non-linear benchmark)

Pipeline:
- Daten laden (train.csv, val.csv)
- Features & Target trennen
- Standardisierung für Logistic Regression
- Modelle trainieren
- Validation-Performance auswerten
- Feature-Weights der Logistic Regression speichern
- Confusion Matrices speichern
- Accuracy & F1 Score ausgeben
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# 1) Projekt-Konfiguration
# ---------------------------------------------------------
@dataclass
class ProjectConfig:
    base_dir: Path = Path(__file__).resolve().parents[1]

    @property
    def processed_data_dir(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.base_dir / "figures"

    @property
    def model_outputs_dir(self) -> Path:
        return self.base_dir / "model_outputs"


# ---------------------------------------------------------
# 2) Modeling Workflow
# ---------------------------------------------------------
class ModelingPipeline:
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg

    # ---------- Load data ----------
    def load_data(self):
        train = pd.read_csv(self.cfg.processed_data_dir / "train.csv", index_col=0)
        val = pd.read_csv(self.cfg.processed_data_dir / "val.csv", index_col=0)

        # target column
        target_col = "target_up"

        # Features = alle numerischen Spalten außer target
        feature_cols = [c for c in train.columns if c not in ["target_up", "future_ret_15m"]]

        X_train = train[feature_cols]
        y_train = train[target_col]

        X_val = val[feature_cols]
        y_val = val[target_col]

        return X_train, y_train, X_val, y_val, feature_cols

    # ---------- Logistic Regression ----------
    def train_logistic_regression(self, X_train, y_train, X_val, y_val, feature_cols):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        log_reg = LogisticRegression(max_iter=500)
        log_reg.fit(X_train_scaled, y_train)

        y_pred_train = log_reg.predict(X_train_scaled)
        y_pred_val = log_reg.predict(X_val_scaled)

        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)

        train_f1 = f1_score(y_train, y_pred_train)
        val_f1 = f1_score(y_val, y_pred_val)

        # Feature weights
        coef = log_reg.coef_[0]
        weights = pd.DataFrame({
            "feature": feature_cols,
            "weight": coef
        }).sort_values("weight", ascending=False)

        self.cfg.model_outputs_dir.mkdir(parents=True, exist_ok=True)
        weights.to_csv(self.cfg.model_outputs_dir / "logistic_regression_feature_weights.csv", index=False)

        print("\n=== Logistic Regression ===")
        print(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

        return log_reg, scaler, weights

    # ---------- Random Forest ----------
    def train_random_forest(self, X_train, y_train, X_val, y_val, feature_cols):
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42
        )
        rf.fit(X_train, y_train)

        y_pred_train = rf.predict(X_train)
        y_pred_val = rf.predict(X_val)

        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)

        train_f1 = f1_score(y_train, y_pred_train)
        val_f1 = f1_score(y_val, y_pred_val)

        # Feature importance speichern
        importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)

        importances.to_csv(self.cfg.model_outputs_dir / "random_forest_feature_importance.csv", index=False)

        print("\n=== Random Forest ===")
        print(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

        return rf, importances

    # ---------- Confusion Matrix ----------
    def save_confusion_matrix(self, y_true, y_pred, model_name):
        self.cfg.figures_dir.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix — {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        plt.savefig(self.cfg.figures_dir / f"cm_{model_name}.png")
        plt.close()

    # ---------- Run whole pipeline ----------
    def run(self):
        X_train, y_train, X_val, y_val, feature_cols = self.load_data()

        # Logistic Regression
        log_reg, scaler, weights = self.train_logistic_regression(X_train, y_train, X_val, y_val, feature_cols)
        self.save_confusion_matrix(y_val, log_reg.predict(scaler.transform(X_val)), "logistic_regression")

        # Random Forest
        rf, importances = self.train_random_forest(X_train, y_train, X_val, y_val, feature_cols)
        self.save_confusion_matrix(y_val, rf.predict(X_val), "random_forest")


# ---------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------
def main():
    cfg = ProjectConfig()
    pipeline = ModelingPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
