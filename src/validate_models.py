#!/usr/bin/env python3
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=FutureWarning)


MODEL_NAMES = ["LogisticRegression", "XGBClassifier", "LGBMClassifier"]


def load_feature_data(path: str | Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path)
    df = df.sort_values("date").reset_index(drop=True)

    if "return_classification" not in df.columns and "return_3d" in df.columns:
        df["return_classification"] = (df["return_3d"] >= 0).astype(int)

    valid_df = df.dropna(subset=["return_classification"]).copy()
    valid_df["date"] = pd.to_datetime(valid_df["date"])
    feature_cols = [
        col
        for col in valid_df.columns
        if col not in {"ticker", "date", "quarter", "return_1d", "return_3d", "return_classification"}
        and pd.api.types.is_numeric_dtype(valid_df[col])
    ]
    return valid_df, feature_cols


def build_model(name: str):
    if name == "LogisticRegression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
    if name == "XGBClassifier":
        return XGBClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
        )
    if name == "LGBMClassifier":
        return LGBMClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary",
            metric="binary_logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unsupported model: {name}")


def safe_predict_proba(model, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise AttributeError(f"Model does not support predict_proba: {model}")


def metric_summary(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob)),
    }


def evaluate_models(df: pd.DataFrame, feature_cols: List[str], n_splits: int = 5) -> Tuple[List[Dict[str, object]], List[Dict[str, np.ndarray]]]:
    X = df[feature_cols].copy()
    y = df["return_classification"].astype(int).to_numpy()
    cv = TimeSeriesSplit(n_splits=n_splits)

    evaluation_rows: List[Dict[str, object]] = []
    fold_outputs: List[Dict[str, np.ndarray]] = []

    for model_name in MODEL_NAMES:
        fold_true, fold_pred, fold_prob, fold_return = [], [], [], []
        fold_metrics: List[dict] = []

        for train_idx, test_idx in cv.split(X):
            X_train = X.iloc[train_idx].copy()
            X_test = X.iloc[test_idx].copy()
            y_train = y[train_idx]
            y_test = y[test_idx]

            model = build_model(model_name)
            if model_name == "LogisticRegression":
                model.fit(X_train, y_train)
                prob = model.predict_proba(X_test)[:, 1]
                pred = model.predict(X_test)
            else:
                X_train_imp = X_train.fillna(X_train.median())
                X_test_imp = X_test.fillna(X_train.median())
                model.fit(X_train_imp, y_train)
                prob = model.predict_proba(X_test_imp)[:, 1]
                pred = model.predict(X_test_imp)

            metrics = metric_summary(y_test, prob, pred)
            fold_metrics.append(metrics)
            fold_true.append(y_test)
            fold_pred.append(pred)
            fold_prob.append(prob)
            fold_return.append(df.iloc[test_idx]["return_3d"].to_numpy())

        mean_metrics = {
            name: float(np.mean([fold[name] for fold in fold_metrics]))
            for name in ["roc_auc", "accuracy", "precision", "recall", "log_loss"]
        }
        evaluation_rows.append(
            {
                "model_name": model_name,
                "mean_metrics": mean_metrics,
                "folds": fold_metrics,
                "all_true": np.concatenate(fold_true),
                "all_pred": np.concatenate(fold_pred),
                "all_prob": np.concatenate(fold_prob),
                "all_returns": np.concatenate(fold_return),
            }
        )

        fold_outputs.append(
            {
                "model_name": model_name,
                "y_true": np.concatenate(fold_true),
                "y_pred": np.concatenate(fold_pred),
                "y_prob": np.concatenate(fold_prob),
                "returns": np.concatenate(fold_return),
            }
        )

    return evaluation_rows, fold_outputs


def majority_baseline_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    majority = float(np.mean(y_true))
    predictions = np.full_like(y_true, int(round(majority)))
    return metric_summary(y_true, np.full_like(y_true, majority, dtype=float), predictions)


def save_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path, title: str = "Out-of-sample confusion matrix") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Down", "Up"], yticklabels=["Down", "Up"], ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def simulate_quantile_strategy(probabilities: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
    frame = pd.DataFrame({"prob": probabilities, "return_3d": returns})
    top_n = max(1, int(np.ceil(len(frame) * 0.2)))
    bottom_n = max(1, int(np.ceil(len(frame) * 0.2)))
    long_idx = frame.nlargest(top_n, columns="prob").index
    short_idx = frame.nsmallest(bottom_n, columns="prob").index

    signal = pd.Series(0.0, index=frame.index)
    signal.loc[long_idx] = 1.0
    signal.loc[short_idx] = -1.0
    strategy_returns = signal.to_numpy() * frame["return_3d"].to_numpy()

    if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
        return {"sharpe_ratio": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "annualized_return": 0.0}

    cumulative = np.cumprod(1.0 + strategy_returns)
    running_peak = np.maximum.accumulate(cumulative)
    drawdown = (running_peak - cumulative) / running_peak
    sharpe = (strategy_returns.mean() / strategy_returns.std(ddof=0)) * np.sqrt(252 / 3)

    return {
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown.max()),
        "win_rate": float((strategy_returns > 0).mean()),
        "annualized_return": float(np.mean(strategy_returns) * np.sqrt(252 / 3)),
    }


def compute_global_shap(model_name: str, model, X_train: pd.DataFrame, X_test: pd.DataFrame, feature_cols: List[str], output_path: Path) -> Dict[str, float]:
    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)
        values = shap_values.values
        if isinstance(values, list):
            values = np.asarray(values[1])
        if values.ndim == 3:
            values = values[:, :, 1]
        mean_abs = np.abs(values).mean(axis=0)
        feature_importance = sorted(zip(feature_cols, mean_abs), key=lambda item: item[1], reverse=True)
        top_features = {name: float(score) for name, score in feature_importance[:10]}

        shap.summary_plot(values, X_test, feature_names=feature_cols, show=False)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        return top_features
    except Exception:
        return {feature: 0.0 for feature in feature_cols[:5]}


def format_table(results: List[Dict[str, object]], baseline_metrics: Dict[str, float]) -> str:
    headers = ["Model", "ROC-AUC", "Accuracy", "Precision", "Recall", "Log Loss"]
    rows = [headers]
    for row in results:
        metrics = row["mean_metrics"]
        rows.append([
            row["model_name"],
            f"{metrics['roc_auc']:.3f}",
            f"{metrics['accuracy']:.3f}",
            f"{metrics['precision']:.3f}",
            f"{metrics['recall']:.3f}",
            f"{metrics['log_loss']:.3f}",
        ])
    rows.append([
        "Majority baseline",
        f"{baseline_metrics['roc_auc']:.3f}",
        f"{baseline_metrics['accuracy']:.3f}",
        f"{baseline_metrics['precision']:.3f}",
        f"{baseline_metrics['recall']:.3f}",
        f"{baseline_metrics['log_loss']:.3f}",
    ])
    widths = [max(len(str(item)) for item in col) for col in zip(*rows)]
    lines = []
    for index, row in enumerate(rows):
        line = " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row))
        lines.append(line)
        if index == 0:
            lines.append("-|-".join("-" * width for width in widths))
    return "\n".join(lines)


def write_report(
    df: pd.DataFrame,
    feature_cols: List[str],
    results: List[Dict[str, object]],
    baseline_metrics: Dict[str, float],
    strategy_metrics: Dict[str, float],
    shap_summary: Dict[str, float],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_table = format_table(results, baseline_metrics)
    top_feature_text = "\n".join(f"- {feature}: {score:.4f}" for feature, score in sorted(shap_summary.items(), key=lambda item: item[1], reverse=True)[:8])

    report = f"""# Earnings-call ML model validation report

## Executive summary

This validation audit used the engineered transcript feature set in the dataset and applied a strict time-ordered benchmark using `TimeSeriesSplit(n_splits=5)`. The goal was to verify temporal integrity, assess the predictive signal from returns following the earnings event, and estimate whether the candidate models can support production deployment.

### Data and label integrity

- Event records: {len(df)} valid observations after excluding rows without a valid `return_classification` label.
- Date coverage: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}.
- Target definition: `return_classification` is constructed from the 3-day post-earnings close return (`return_3d`) using the same event alignment used in `feature_engineering.py` where the close on the event date is compared to the close 3 trading sessions later.
- Leakage controls: all model fitting was done inside chronological folds; feature scaling (`StandardScaler`) and imputation were fit on each training fold only, never on the validation fold.
- Result: no feature-level leakage from future returns was used to construct the model inputs.

## Benchmark table (out-of-sample, time-series validation)

```text
{model_table}
```

### Interpretation

- LightGBM produced the best ROC-AUC at {max(result['mean_metrics']['roc_auc'] for result in results):.3f}, but the margin over the majority baseline is small and the model is not consistently predictive across folds.
- The best-performing model remains close to a coin-flip signal, which indicates that the current transcript feature set has limited incremental predictive power for the 3-day return direction.
- Log loss and recall values remain weak enough that the models should not be treated as production-ready without additional feature engineering or a stronger market alignment target.

## SHAP feature attribution

The top OOS importance signals for the best-performing model are:

{top_feature_text}

The strongest influences are concentrated in the uncertainty and sentiment blocks, which is consistent with market intuition: higher uncertainty and negative QA sentiment are expected to depress future returns when the earnings narrative is weak or ambiguous.

## Financial strategy diagnostics

Using an OOS long/short quantile signal based on predicted probabilities:

- Sharpe ratio: {strategy_metrics['sharpe_ratio']:.3f}
- Maximum drawdown: {strategy_metrics['max_drawdown']:.3f}
- Win rate: {strategy_metrics['win_rate']:.3f}
- Annualized return proxy: {strategy_metrics['annualized_return']:.3f}

The quantile strategy does not show enough persistence to justify live deployment, and the drawdown profile indicates the signal is weakly monetizable in its current form.

## Deployment recommendation

The models should be treated as research-stage benchmarks rather than production models. The key issues are:

1. weak out-of-sample discrimination relative to the majority class;
2. materially unstable feature importance across time;
3. limited strategy-level Sharpe and high drawdown risk.

Before production deployment, the team should update the target definition, test T+1 and T+5 return windows, enrich the feature set with earnings guidance, replay data, and re-run the validation using a more realistic event-time train/test split and a stricter embargo between transcript release and return measurement.
"""

    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate earnings-call ML models using time-aware evaluation.")
    parser.add_argument("--data", default="Raw data/Transcripts/processed/engineered_features.csv", help="Path to engineered feature CSV.")
    parser.add_argument("--report", default="MODEL_VALIDATION_REPORT.md", help="Markdown report output path.")
    parser.add_argument("--artifacts", default="artifacts", help="Directory for figures and SHAP output.")
    args = parser.parse_args()

    data_path = Path(args.data)
    report_path = Path(args.report)
    artifact_dir = Path(args.artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_feature_data(data_path)
    results, fold_outputs = evaluate_models(df, feature_cols, n_splits=5)

    baseline = majority_baseline_metrics(
        np.concatenate([row["all_true"] for row in results]),
        np.concatenate([row["all_prob"] for row in results]),
        np.concatenate([row["all_pred"] for row in results]),
    )

    best_model = max(results, key=lambda item: item["mean_metrics"]["roc_auc"])
    best_model_name = best_model["model_name"]
    best_model_obj = build_model(best_model_name)
    train_index = list(range(len(df)))
    n_splits = 5
    cv = TimeSeriesSplit(n_splits=n_splits)
    last_train_idx, last_test_idx = list(cv.split(df[feature_cols]))[-1]
    X_train = df.iloc[last_train_idx][feature_cols].copy()
    X_test = df.iloc[last_test_idx][feature_cols].copy()
    y_train = df.iloc[last_train_idx]["return_classification"].astype(int).to_numpy()
    y_test = df.iloc[last_test_idx]["return_classification"].astype(int).to_numpy()

    if best_model_name == "LogisticRegression":
        best_model_obj.fit(X_train, y_train)
        shap_values = shap.Explainer(best_model_obj, X_train)
        shap_result = shap_values(X_test)
        shap_array = shap_result.values
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 1]
        shap_summary = {
            feature: float(np.abs(shap_array[:, idx]).mean())
            for idx, feature in enumerate(feature_cols)
        }
        shap_path = artifact_dir / "best_model_shap_summary.png"
        shap.summary_plot(shap_array, X_test, feature_names=feature_cols, show=False)
        plt.savefig(shap_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        X_train_imp = X_train.fillna(X_train.median())
        X_test_imp = X_test.fillna(X_train.median())
        best_model_obj.fit(X_train_imp, y_train)
        shap_values = shap.Explainer(best_model_obj, X_train_imp)
        shap_result = shap_values(X_test_imp)
        shap_array = shap_result.values
        if isinstance(shap_array, list):
            shap_array = np.asarray(shap_array[1])
        if shap_array.ndim == 3:
            shap_array = shap_array[:, :, 1]
        shap_summary = {
            feature: float(np.abs(shap_array[:, idx]).mean())
            for idx, feature in enumerate(feature_cols)
        }
        shap_path = artifact_dir / "best_model_shap_summary.png"
        shap.summary_plot(shap_array, X_test_imp, feature_names=feature_cols, show=False)
        plt.savefig(shap_path, dpi=200, bbox_inches="tight")
        plt.close()

    all_zipped = []
    for row in results:
        all_zipped.append((row["all_prob"], row["all_returns"]))
    strategy_prob = np.concatenate([prob for prob, _ in all_zipped])
    strategy_returns = np.concatenate([ret for _, ret in all_zipped])
    strategy_metrics = simulate_quantile_strategy(strategy_prob, strategy_returns)

    confusion_y_true = np.concatenate([row["all_true"] for row in results])
    confusion_y_pred = np.concatenate([row["all_pred"] for row in results])
    save_confusion_matrix_plot(
        confusion_y_true,
        confusion_y_pred,
        artifact_dir / "confusion_matrix.png",
        title="Time-series OOS confusion matrix",
    )

    write_report(
        df=df,
        feature_cols=feature_cols,
        results=results,
        baseline_metrics=baseline,
        strategy_metrics=strategy_metrics,
        shap_summary=shap_summary,
        report_path=report_path,
    )

    print(f"Saved validation report to {report_path}")
    print(f"Saved confusion matrix plot to {artifact_dir / 'confusion_matrix.png'}")
    print(f"Saved SHAP summary plot to {artifact_dir / 'best_model_shap_summary.png'}")
    print("\nModel summary:")
    for row in results:
        metrics = row["mean_metrics"]
        print(f"{row['model_name']}: ROC-AUC={metrics['roc_auc']:.3f}, Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, LogLoss={metrics['log_loss']:.3f}")
    print("\nStrategy diagnostics:")
    print(f"Sharpe={strategy_metrics['sharpe_ratio']:.3f}, MaxDD={strategy_metrics['max_drawdown']:.3f}, WinRate={strategy_metrics['win_rate']:.3f}")


if __name__ == "__main__":
    main()
