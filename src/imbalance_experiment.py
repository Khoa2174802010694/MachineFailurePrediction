import os
import json
import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

TARGET = "Machine failure"
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5


def load_and_prepare_data():
    print("\\n[1] Loading dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found.")

    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\\nTarget distribution:")
    print(df[TARGET].value_counts().sort_index())
    print(f"Failure rate: {df[TARGET].mean() * 100:.2f}%")

    # Remove identifiers and failure-mode labels to avoid leakage.
    columns_to_drop = [
        "UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"
    ]
    X = df.drop(columns=columns_to_drop + [TARGET], errors="ignore")
    y = df[TARGET]

    categorical_features = X.select_dtypes(
        include=["object", "category"]
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
        X, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y,
    )

    print(f"\\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Training failure rate: {y_train.mean() * 100:.2f}%")
    print(f"Testing failure rate: {y_test.mean() * 100:.2f}%")
    return X_train, X_test, y_train, y_test, preprocessor


def build_experiments(preprocessor):
    # Only the imbalance-handling strategy changes.
    rf_params = {
        "n_estimators": 200,
        "max_depth": 12,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    return {
        "RF - No Balancing": ImbPipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", RandomForestClassifier(
                **rf_params, class_weight=None
            )),
        ]),
        "RF - Class Weight Balanced": ImbPipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", RandomForestClassifier(
                **rf_params, class_weight="balanced"
            )),
        ]),
        "RF - Random Oversampling": ImbPipeline([
            ("preprocessor", clone(preprocessor)),
            ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
            ("classifier", RandomForestClassifier(
                **rf_params, class_weight=None
            )),
        ]),
    }


def run_experiments(X_train, X_test, y_train, y_test, preprocessor):
    print("\\n[2] Running imbalance-handling experiments...")
    experiments = build_experiments(preprocessor)
    cv = StratifiedKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    scoring = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "average_precision": "average_precision",
    }

    rows = []
    fitted_models = {}

    for name, pipeline in experiments.items():
        print(f"\\nProcessing: {name}")

        # Preprocessing and oversampling are inside the pipeline.
        # Each CV fold learns them using only its training partition.
        cv_results = cross_validate(
            pipeline, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=1,
            return_train_score=False,
        )

        cv_f1_mean = cv_results["test_f1"].mean()
        cv_f1_std = cv_results["test_f1"].std()
        cv_precision_mean = cv_results["test_precision"].mean()
        cv_recall_mean = cv_results["test_recall"].mean()
        cv_ap_mean = cv_results["test_average_precision"].mean()

        print(
            f"CV F1: {cv_f1_mean:.4f} +/- {cv_f1_std:.4f} | "
            f"CV Precision: {cv_precision_mean:.4f} | "
            f"CV Recall: {cv_recall_mean:.4f} | "
            f"CV PR-AUC: {cv_ap_mean:.4f}"
        )

        # Fit on all training data; evaluate once on untouched test data.
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Experiment": name,
            "CV F1 mean": cv_f1_mean,
            "CV F1 std": cv_f1_std,
            "CV Precision mean": cv_precision_mean,
            "CV Recall mean": cv_recall_mean,
            "CV PR-AUC mean": cv_ap_mean,
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Test Precision": precision_score(y_test, y_pred, zero_division=0),
            "Test Recall": recall_score(y_test, y_pred, zero_division=0),
            "Test F1": f1_score(y_test, y_pred, zero_division=0),
            "Test ROC-AUC": roc_auc_score(y_test, y_prob),
            "Test PR-AUC": average_precision_score(y_test, y_prob),
        }
        rows.append(metrics)
        fitted_models[name] = pipeline

        print(
            f"Test Accuracy:  {metrics['Test Accuracy']:.4f}\\n"
            f"Test Precision: {metrics['Test Precision']:.4f}\\n"
            f"Test Recall:    {metrics['Test Recall']:.4f}\\n"
            f"Test F1:        {metrics['Test F1']:.4f}\\n"
            f"Test ROC-AUC:   {metrics['Test ROC-AUC']:.4f}\\n"
            f"Test PR-AUC:    {metrics['Test PR-AUC']:.4f}"
        )

    results = pd.DataFrame(rows).sort_values(
        by="CV F1 mean", ascending=False
    ).reset_index(drop=True)
    return results, fitted_models


def save_experiment_results(results, fitted_models, X_test, y_test):
    print("\\n[3] Saving experiment results...")
    results_path = os.path.join(RESULT_DIR, "imbalance_experiment.csv")
    results.to_csv(results_path, index=False)

    best_name = results.iloc[0]["Experiment"]
    best_model = fitted_models[best_name]
    y_pred = best_model.predict(X_test)

    report_path = os.path.join(
        RESULT_DIR, "imbalance_experiment_report.txt"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Best experiment by CV F1 mean: {best_name}\\n\\n")
        f.write(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(
        cm,
        index=["Actual Normal", "Actual Failure"],
        columns=["Predicted Normal", "Predicted Failure"],
    ).to_csv(os.path.join(
        RESULT_DIR, "imbalance_experiment_confusion_matrix.csv"
    ))

    # Store separately. The Streamlit app's best_model.pkl is untouched.
    joblib.dump(
        best_model,
        os.path.join(RESULT_DIR, "best_imbalance_experiment.pkl"),
    )

    metadata = {
        "best_experiment": best_name,
        "selection_criterion": (
            "Mean F1-score from stratified 5-fold cross-validation "
            "on training data"
        ),
        "cross_validation_folds": N_SPLITS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "note": (
            "Separate research experiment; models/best_model.pkl "
            "was not overwritten."
        ),
    }
    with open(
        os.path.join(RESULT_DIR, "imbalance_experiment_metadata.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print(f"\\nBest experiment by CV F1 mean: {best_name}")
    print("\\nExperiment comparison:")
    print(results.to_string(index=False))
    print("\\nSaved experiment files in results/.")
    print("Original models/best_model.pkl was NOT changed.")


def main():
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare_data()
    results, fitted_models = run_experiments(
        X_train, X_test, y_train, y_test, preprocessor
    )
    save_experiment_results(results, fitted_models, X_test, y_test)
    print("\\n========================================")
    print("IMBALANCE EXPERIMENT COMPLETED!")
    print("========================================")


if __name__ == "__main__":
    main()
