import json
import unittest
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from src.inference import FEATURES, predict_with_threshold, validate_input_data

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore", category=UserWarning)
        cls.model = joblib.load(ROOT / "results" / "threshold_tuned_random_forest.pkl")
        with open(ROOT / "results" / "threshold_experiment_metadata.json", encoding="utf-8") as file:
            cls.metadata = json.load(file)
        cls.threshold = float(cls.metadata["selected_threshold"])
        cls.dataset = pd.read_csv(ROOT / "data" / "ai4i2020.csv")

    def test_valid_csv_data(self):
        valid, result = validate_input_data(self.dataset[FEATURES].head(5))
        self.assertTrue(valid, result)
        self.assertEqual(list(result.columns), FEATURES)

    def test_reject_missing_column(self):
        valid, message = validate_input_data(self.dataset[FEATURES].head(2).drop(columns=["Torque [Nm]"]))
        self.assertFalse(valid)
        self.assertIn("Torque [Nm]", message)

    def test_reject_invalid_machine_type(self):
        sample = self.dataset[FEATURES].head(2).copy()
        sample.loc[sample.index[0], "Type"] = "X"
        valid, message = validate_input_data(sample)
        self.assertFalse(valid)
        self.assertIn("Type", message)

    def test_reject_out_of_range_value(self):
        sample = self.dataset[FEATURES].head(2).copy()
        sample.loc[sample.index[0], "Tool wear [min]"] = 999
        valid, message = validate_input_data(sample)
        self.assertFalse(valid)
        self.assertIn("Tool wear", message)

    def test_threshold_applied_to_probabilities(self):
        sample = self.dataset[FEATURES].head(25)
        predictions, probabilities = predict_with_threshold(self.model, sample, self.threshold)
        self.assertEqual(len(predictions), len(sample))
        self.assertTrue(np.isfinite(probabilities).all())
        np.testing.assert_array_equal(predictions, (probabilities >= self.threshold).astype(int))

    def test_threshold_metrics_match_saved_test_results(self):
        target = self.dataset["Machine failure"]
        X = self.dataset[FEATURES]
        X_train, X_test, y_train, y_test = train_test_split(
            X, target, test_size=0.2, random_state=42, stratify=target
        )
        probabilities = self.model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        expected = pd.read_csv(ROOT / "results" / "threshold_test_comparison.csv").iloc[1]
        self.assertAlmostEqual(accuracy_score(y_test, predictions), expected["Accuracy"], places=6)
        self.assertAlmostEqual(precision_score(y_test, predictions, zero_division=0), expected["Precision"], places=6)
        self.assertAlmostEqual(recall_score(y_test, predictions, zero_division=0), expected["Recall"], places=6)
        self.assertAlmostEqual(f1_score(y_test, predictions, zero_division=0), expected["F1"], places=6)


if __name__ == "__main__":
    unittest.main()
