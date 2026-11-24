import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

TARGET = "Machine failure"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5

# Pre-declared threshold candidates. The test set is NOT used
# to choose the threshold.
THRESHOLDS = np.round(np.arange(0.10, 0.91, 0.05), 2)


# =========================================================
# 2. LOAD AND PREPARE DATA
# =========================================================

def load_and_prepare_data():
    print("\n[1] Loading dataset...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found.")

    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nTarget distribution:")
    print(df[TARGET].value_counts().sort_index())
    print(f"Failure rate: {df[TARGET].mean() * 100:.2f}%")

    # Exclude identifiers and failure-mode labels.
    columns_to_drop = [
        "UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"
    ]
    X = df.drop(columns=columns_to_drop + [TARGET], errors="ignore")
    y = df[TARGET]

    categorical_features = X.select_dtypes(
        include=["object", "category", "str"]
    ).columns.tolist()
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training failure rate: {y_train.mean() * 100:.2f}%")
    print(f"Testing failure rate: {y_test.mean() * 100:.2f}%")

    return X_train, X_test, y_train, y_test, preprocessor


# =========================================================
# 3. BUILD RANDOM FOREST
# =========================================================

def build_model(preprocessor):
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier),
    ])


# =========================================================
# 4. SELECT THRESHOLD USING OUT-OF-FOLD TRAIN PROBABILITIES
# =========================================================

def select_threshold(X_train, y_train, preprocessor):
    print("\n[2] Generating out-of-fold probabilities on training data...")

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = build_model(preprocessor)

    # Each training row receives a probability from a model that
    # did not train on that row.
    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    rows = []
    for threshold in THRESHOLDS:
        oof_predictions = (oof_probabilities >= threshold).astype(int)

        rows.append({
            "Threshold": float(threshold),
            "OOF Precision": precision_score(
                y_train, oof_predictions, zero_division=0
            ),
            "OOF Recall": recall_score(
                y_train, oof_predictions, zero_division=0
            ),
            "OOF F1": f1_score(
                y_train, oof_predictions, zero_division=0
            ),
            "OOF Accuracy": accuracy_score(
                y_train, oof_predictions
            ),
        })

    threshold_results = pd.DataFrame(rows)

    # Threshold selection is based only on training OOF F1.
    # If there is a tie, choose the higher threshold to avoid
    # unnecessarily increasing false positives.
    best_f1 = threshold_results["OOF F1"].max()
    tied = threshold_results[
        np.isclose(threshold_results["OOF F1"], best_f1)
    ]
    best_row = tied.sort_values(
        "Threshold", ascending=False
    ).iloc[0]

    best_threshold = float(best_row["Threshold"])

    print("\nOOF threshold comparison:")
    print(threshold_results.to_string(index=False))
    print(f"\nSelected threshold: {best_threshold:.2f}")
    print(f"Selected OOF F1: {best_row['OOF F1']:.4f}")

    return best_threshold, threshold_results


# =========================================================
# 5. EVALUATE ON UNTOUCHED TEST SET
# =========================================================

def evaluate_on_test(
    X_train, X_test, y_train, y_test, preprocessor, best_threshold
):
    print("\n[3] Fitting final model and evaluating on untouched test data...")

    final_model = build_model(preprocessor)
    final_model.fit(X_train, y_train)

    test_probabilities = final_model.predict_proba(X_test)[:, 1]

    predictions_default = (test_probabilities >= 0.50).astype(int)
    predictions_tuned = (
        test_probabilities >= best_threshold
    ).astype(int)

    rows = []

    for name, predictions in [
        ("Default threshold (0.50)", predictions_default),
        (f"Tuned threshold ({best_threshold:.2f})", predictions_tuned),
    ]:
        rows.append({
            "Configuration": name,
            "Threshold": (
                0.50 if name.startswith("Default") else best_threshold
            ),
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_test, test_probabilities),
            "PR-AUC": average_precision_score(
                y_test, test_probabilities
            ),
        })

    test_results = pd.DataFrame(rows)

    print("\nTest-set comparison:")
    print(test_results.to_string(index=False))

    return final_model, test_results, predictions_default, predictions_tuned


# =========================================================
# 6. SAVE RESULTS WITHOUT OVERWRITING THE APP MODEL
# =========================================================

def save_results(
    best_threshold,
    threshold_results,
    final_model,
    test_results,
    y_test,
    predictions_default,
    predictions_tuned,
):
    print("\n[4] Saving threshold experiment results...")

    threshold_results.to_csv(
        os.path.join(RESULT_DIR, "threshold_oof_comparison.csv"),
        index=False,
    )

    test_results.to_csv(
        os.path.join(RESULT_DIR, "threshold_test_comparison.csv"),
        index=False,
    )

    cm_default = confusion_matrix(y_test, predictions_default)
    cm_tuned = confusion_matrix(y_test, predictions_tuned)

    cm_df = pd.DataFrame(
        [
            [
                cm_default[0, 0], cm_default[0, 1],
                cm_default[1, 0], cm_default[1, 1],
            ],
            [
                cm_tuned[0, 0], cm_tuned[0, 1],
                cm_tuned[1, 0], cm_tuned[1, 1],
            ],
        ],
        columns=[
            "TN", "FP", "FN", "TP"
        ],
        index=[
            "Default threshold 0.50",
            f"Tuned threshold {best_threshold:.2f}",
        ],
    )
    cm_df.to_csv(
        os.path.join(RESULT_DIR, "threshold_confusion_matrices.csv")
    )

    # Save separately; do not replace models/best_model.pkl.
    joblib.dump(
        final_model,
        os.path.join(RESULT_DIR, "threshold_tuned_random_forest.pkl"),
    )

    tuned_row = test_results.iloc[1]
    report = classification_report(
        y_test,
        (predictions_tuned).astype(int),
        zero_division=0,
    )

    with open(
        os.path.join(RESULT_DIR, "threshold_experiment_report.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("Threshold tuning experiment\n")
        f.write("===========================\n\n")
        f.write(
            "Threshold selected using out-of-fold probabilities "
            "from training data only.\n"
        )
        f.write(f"Selected threshold: {best_threshold:.2f}\n\n")
        f.write("Test-set comparison:\n")
        f.write(test_results.to_string(index=False))
        f.write("\n\nClassification report at tuned threshold:\n")
        f.write(report)

    metadata = {
        "model": "RandomForestClassifier",
        "class_weight": "balanced",
        "n_estimators": 200,
        "max_depth": 12,
        "threshold_selection": "Maximum OOF F1 on training data",
        "selected_threshold": best_threshold,
        "cross_validation_folds": N_SPLITS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "app_model_overwritten": False,
    }

    with open(
        os.path.join(RESULT_DIR, "threshold_experiment_metadata.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("\nSaved files in results/:")
    print("- threshold_oof_comparison.csv")
    print("- threshold_test_comparison.csv")
    print("- threshold_confusion_matrices.csv")
    print("- threshold_experiment_report.txt")
    print("- threshold_experiment_metadata.json")
    print("- threshold_tuned_random_forest.pkl")
    print("\nOriginal models/best_model.pkl was NOT changed.")


# =========================================================
# 7. MAIN
# =========================================================

def main():
    X_train, X_test, y_train, y_test, preprocessor = (
        load_and_prepare_data()
    )

    best_threshold, threshold_results = select_threshold(
        X_train, y_train, preprocessor
    )

    (
        final_model,
        test_results,
        predictions_default,
        predictions_tuned,
    ) = evaluate_on_test(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        best_threshold,
    )

    save_results(
        best_threshold,
        threshold_results,
        final_model,
        test_results,
        y_test,
        predictions_default,
        predictions_tuned,
    )

    print("\n========================================")
    print("THRESHOLD EXPERIMENT COMPLETED!")
    print("========================================")


if __name__ == "__main__":
    main()
