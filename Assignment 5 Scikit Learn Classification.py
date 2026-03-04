import os
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import csv

output_file = open("breast_cancer_classification_results.csv", "w", newline='')
writer = csv.writer(output_file)

# Write header row
writer.writerow(["Model", "Accuracy", "F1 (macro)", "Precision (macro)", "Recall (macro)"])

class BreastCancerModelRunner:
    """
    Encapsulates loading data, training multiple classifiers,
    evaluating them, and reporting metrics.
    """

    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.data_bunch = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.models = {}
        self.results = {}

    def load_data(self):
        """
        Load the built-in breast cancer dataset from scikit-learn
        and convert it to a pandas DataFrame for convenience.
        """
        self.data_bunch = load_breast_cancer()
        self.X = pd.DataFrame(
            self.data_bunch.data,
            columns=self.data_bunch.feature_names
        )
        self.y = pd.Series(self.data_bunch.target, name="target")
        self.feature_names = self.data_bunch.feature_names
        self.target_names = self.data_bunch.target_names

    def train_test_split(self):
        """
        Split the data into training and testing sets using the
        standard scikit-learn approach.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
            stratify=self.y,
        )

    def build_models(self):
        """
        Define three different classification models with some
        parameter optimization for extra credit.
        """

        # 1. Logistic Regression (with regularization and scaling)
        log_reg_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        solver="lbfgs",
                        max_iter=500,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        # 2. Random Forest (tuned number of trees and depth)
        rf_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            criterion="gini",
            random_state=self.random_state,
            n_jobs=-1,
        )

        # 3. Support Vector Machine (RBF kernel with tuned C and gamma)
        svm_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=5.0,
                        gamma="scale",
                        probability=True,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        self.models = {
            "LogisticRegression": log_reg_pipeline,
            "RandomForest": rf_clf,
            "SVM_RBF": svm_pipeline,
        }

    def fit_models(self):
        """
        Train all models on the training data.
        """
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)

    def evaluate_models(self):
        """
        Evaluate each model using multiple metrics and store results.
        """
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)

            acc = accuracy_score(self.y_test, y_pred)
            f1_macro = f1_score(self.y_test, y_pred, average="macro")
            precision_macro = precision_score(
                self.y_test, y_pred, average="macro", zero_division=0
            )
            recall_macro = recall_score(
                self.y_test, y_pred, average="macro", zero_division=0
            )
            cm = confusion_matrix(self.y_test, y_pred)
            report = classification_report(
                self.y_test, y_pred, target_names=self.target_names, zero_division=0
            )

            self.results[name] = {
                "accuracy": acc,
                "f1_macro": f1_macro,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "confusion_matrix": cm,
                "classification_report": report,
            }

    def print_results(self):
        """
        Print a summary of metrics for each model.
        """
        for name, metrics_dict in self.results.items():
            print("=" * 60)
            print(f"Model: {name}")
            print("-" * 60)
            print(f"Accuracy:          {metrics_dict['accuracy']:.4f}")
            print(f"F1 (macro):        {metrics_dict['f1_macro']:.4f}")
            print(f"Precision (macro): {metrics_dict['precision_macro']:.4f}")
            print(f"Recall (macro):    {metrics_dict['recall_macro']:.4f}")
            print("Confusion Matrix:")
            print(metrics_dict["confusion_matrix"])
            print("Classification Report:")
            print(metrics_dict["classification_report"])

    def get_best_model_by_metric(self, metric="accuracy"):
        """
        Return the model name and metrics for the best model
        according to a chosen metric.
        """
        best_name = None
        best_value = -np.inf

        for name, metrics_dict in self.results.items():
            value = metrics_dict.get(metric, -np.inf)
            if value > best_value:
                best_value = value
                best_name = name

        return best_name, self.results.get(best_name, {})


def main():
    # Optional: ensure output directory exists (example use of os)
    os.makedirs("outputs", exist_ok=True)

    def print_both(text):
        print(text)
        output_file.write(text + "\n")

    runner = BreastCancerModelRunner(test_size=0.2, random_state=42)
    runner.load_data()
    runner.train_test_split()
    runner.build_models()
    runner.fit_models()
    runner.evaluate_models()

    for name, metrics_dict in runner.results.items():
        print_both(f"Model: {name}")
        print_both(f"Accuracy: {metrics_dict['accuracy']:.4f}")
        print_both(f"F1 (macro): {metrics_dict['f1_macro']:.4f}")
        print_both(f"Precision (macro): {metrics_dict['precision_macro']:.4f}")
        print_both(f"Recall (macro): {metrics_dict['recall_macro']:.4f}")
        print_both(str(metrics_dict["confusion_matrix"]))
        print_both(metrics_dict["classification_report"])



    best_name, best_metrics = runner.get_best_model_by_metric(metric="accuracy")
    print("=" * 60)
    print(f"Best model by accuracy: {best_name}")
    print(f"Accuracy: {best_metrics.get('accuracy', float('nan')):.4f}")
    print(f"F1 (macro): {best_metrics.get('f1_macro', float('nan')):.4f}")
    print(f"Precision (macro): {best_metrics.get('precision_macro', float('nan')):.4f}")
    print(f"Recall (macro): {best_metrics.get('recall_macro', float('nan')):.4f}")


if __name__ == "__main__":
    main()

output_file.close()