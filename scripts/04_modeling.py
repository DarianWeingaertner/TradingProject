# scripts/04_modeling.py
"""
Step 04 — Modeling (SPY 1Min + GLD 1Min Features)
------------------------------------------------
Modelle:
- Logistic Regression
- Random Forest

Features kommen aus train.csv/val.csv (inkl. GLD-Minuten-Features und Cross-Features).

Outputs:
- model_outputs/{SYMBOL}_logreg_feature_weights.csv
- model_outputs/{SYMBOL}_rf_feature_importance.csv
- model_outputs/{SYMBOL}_feature_group_summary.csv
- figures/cm_{SYMBOL}_logistic_regression.png
- figures/cm_{SYMBOL}_random_forest.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
    primary_symbol: str = "SPY"
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
# 2) Modeling Pipeline
# ---------------------------------------------------------
class ModelingPipeline:
    def __init__(self, cfg: ProjectConfig):
        self.cfg = cfg

    # ---------- Load data ----------
    def load_data(self):
        train_path = self.cfg.processed_data_dir / "train.csv"
        val_path = self.cfg.processed_data_dir / "val.csv"

        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(
                "train.csv / val.csv nicht gefunden. Bitte zuerst scripts/03_data_preparation.py laufen lassen."
            )

        train = pd.read_csv(train_path, index_col=0)
        val = pd.read_csv(val_path, index_col=0)

        target_col = "target_up"

        # Alles außer target und future_ret_* ist Feature
        feature_cols = [
            c for c in train.columns
            if c != target_col and not c.startswith("future_ret_")
        ]

        X_train = train[feature_cols]
        y_train = train[target_col]

        X_val = val[feature_cols]
        y_val = val[target_col]

        # Safety checks
        if X_train.isnull().any().any() or X_val.isnull().any().any():
            raise ValueError("❌ NaNs in Features gefunden. Bitte DataPreparation prüfen (dropna).")

        return X_train, y_train, X_val, y_val, feature_cols

    # ---------- Helper: Feature Groups ----------
    @staticmethod
    def feature_group(feature_name: str) -> str:
        # passt zu Script 03 (1Min+1Min)
        if feature_name.startswith("spy_"):
            return "SPY"
        if feature_name.startswith("gld_"):
            return "GLD"
        # Cross-Features aus Script 03
        if feature_name.startswith(("ret_spy_minus_gld", "vol_ratio_spy_gld", "price_ratio_spy_gld")):
            return "CROSS"
        if feature_name in {"hour", "minute_of_day", "minute_of_day_norm"}:
            return "TIME"
        return "OTHER"

    def save_feature_group_summary(self, weights_df: pd.DataFrame, importances_df: pd.DataFrame):
        """
        Aggregiert Beiträge nach Gruppen.
        - Logistic Regression: Sum(|weight|) pro Gruppe
        - Random Forest: Sum(importance) pro Gruppe
        """
        out_dir = self.cfg.model_outputs_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        w = weights_df.copy()
        w["group"] = w["feature"].apply(self.feature_group)
        w_group = w.groupby("group")["weight"].apply(lambda s: s.abs().sum()).reset_index()
        w_group = w_group.rename(columns={"weight": "logreg_sum_abs_weight"})

        imp = importances_df.copy()
        imp["group"] = imp["feature"].apply(self.feature_group)
        imp_group = imp.groupby("group")["importance"].sum().reset_index()
        imp_group = imp_group.rename(columns={"importance": "rf_sum_importance"})

        summary = pd.merge(w_group, imp_group, on="group", how="outer").fillna(0.0)
        summary = summary.sort_values(["rf_sum_importance", "logreg_sum_abs_weight"], ascending=False)

        out_path = out_dir / f"{self.cfg.primary_symbol}_feature_group_summary.csv"
        summary.to_csv(out_path, index=False)
        print(f"💾 Feature-Group Summary gespeichert: {out_path}")

    # ---------- Logistic Regression ----------
    def train_logistic_regression(self, X_train, y_train, X_val, y_val, feature_cols):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LogisticRegression(max_iter=800)
        model.fit(X_train_scaled, y_train)

        y_pred_train = model.predict(X_train_scaled)
        y_pred_val = model.predict(X_val_scaled)

        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)

        train_f1 = f1_score(y_train, y_pred_train)
        val_f1 = f1_score(y_val, y_pred_val)

        weights = pd.DataFrame({"feature": feature_cols, "weight": model.coef_[0]}).sort_values(
            "weight", ascending=False
        )

        self.cfg.model_outputs_dir.mkdir(parents=True, exist_ok=True)
        out = self.cfg.model_outputs_dir / f"{self.cfg.primary_symbol}_logreg_feature_weights.csv"
        weights.to_csv(out, index=False)

        print(f"\n=== Logistic Regression ({self.cfg.primary_symbol}) ===")
        print(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"Train F1:       {train_f1:.4f}, Val F1:       {val_f1:.4f}")
        print(f"💾 Weights gespeichert: {out}")
        print("Top-10 Weights:")
        print(weights.head(10).to_string(index=False))

        return model, scaler, weights

    # ---------- Random Forest ----------
    def train_random_forest(self, X_train, y_train, X_val, y_val, feature_cols):
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        y_pred_train = rf.predict(X_train)
        y_pred_val = rf.predict(X_val)

        train_acc = accuracy_score(y_train, y_pred_train)
        val_acc = accuracy_score(y_val, y_pred_val)

        train_f1 = f1_score(y_train, y_pred_train)
        val_f1 = f1_score(y_val, y_pred_val)

        importances = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_}).sort_values(
            "importance", ascending=False
        )

        self.cfg.model_outputs_dir.mkdir(parents=True, exist_ok=True)
        out = self.cfg.model_outputs_dir / f"{self.cfg.primary_symbol}_rf_feature_importance.csv"
        importances.to_csv(out, index=False)

        print(f"\n=== Random Forest ({self.cfg.primary_symbol}) ===")
        print(f"Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}")
        print(f"Train F1:       {train_f1:.4f}, Val F1:       {val_f1:.4f}")
        print(f"💾 Importances gespeichert: {out}")
        print("Top-10 Importances:")
        print(importances.head(10).to_string(index=False))

        return rf, importances

    # ---------- Confusion Matrix ----------
    def save_confusion_matrix(self, y_true, y_pred, model_name: str):
        self.cfg.figures_dir.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix — {model_name} ({self.cfg.primary_symbol})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        out = self.cfg.figures_dir / f"cm_{self.cfg.primary_symbol}_{model_name}.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"📊 Confusion Matrix gespeichert: {out}")

    # ---------- Run ----------
    def run(self):
        X_train, y_train, X_val, y_val, feature_cols = self.load_data()

        logreg, scaler, weights = self.train_logistic_regression(X_train, y_train, X_val, y_val, feature_cols)
        self.save_confusion_matrix(y_val, logreg.predict(scaler.transform(X_val)), "logistic_regression")

        rf, importances = self.train_random_forest(X_train, y_train, X_val, y_val, feature_cols)
        self.save_confusion_matrix(y_val, rf.predict(X_val), "random_forest")

        # Gruppensummary (zeigt schnell, ob GLD-Minutenfeatures/Cross überhaupt ziehen)
        self.save_feature_group_summary(weights, importances)


def main():
    cfg = ProjectConfig(primary_symbol="SPY")
    ModelingPipeline(cfg).run()


if __name__ == "__main__":
    main()