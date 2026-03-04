# B211-Assignment_5
# Model Performance Summary 
Logistic Regression and the SVM with an RBF kernel were the top-performing models in this project, each achieving 98.25% accuracy along with identical macro‑averaged F1, precision, and recall scores of 0.9812. Both models misclassified only two samples in total, demonstrating highly balanced performance across malignant and benign cases. Although the script lists Logistic Regression as the “best” model, this is only because it appears first in the evaluation order—its performance is statistically tied with the SVM. In contrast, the Random Forest model performed well but lagged behind, reaching 94.74% accuracy and a 0.9435 macro F1, with six total misclassifications. Logistic Regression offers strong interpretability and fast training, while the SVM provides a flexible non‑linear decision boundary that captures more complex patterns. Random Forest remains useful for feature importance insights, but its lower precision and recall make it less competitive in this comparison.

# Breast Cancer Classification with Scikit-Learn

## Project purpose

This project uses the built-in **breast cancer dataset** from Scikit-Learn to explore how different machine learning classification algorithms can support early detection of breast cancer. Over the past 30 years, breast cancer mortality has dropped significantly, in part due to earlier detection and better risk stratification. Here, I implement and compare three supervised learning models to classify tumors as **malignant** or **benign**, using standard training/testing splits and multiple evaluation metrics.

The project is implemented in Python using `os`, `pandas`, `numpy`, `scipy`, and several modules from `scikit-learn`.

---

## Data and preprocessing

- **Dataset:** `sklearn.datasets.load_breast_cancer`
- **Features:** 30 numeric features describing cell nuclei (e.g., mean radius, texture, perimeter).
- **Target:** Binary label (`0` = malignant, `1` = benign).
- **Split:** `train_test_split` with `test_size=0.2`, `random_state=42`, and `stratify=y` to preserve class balance.
- **Scaling:** For linear and SVM models, `StandardScaler` is applied via a `Pipeline` to standardize features.

---

## Class design and implementation

### `BreastCancerModelRunner`

This class encapsulates the full workflow: loading data, splitting, building models, training, evaluating, and summarizing results.

**Attributes:**

- **`test_size`** (`float`): Fraction of data reserved for testing (default `0.2`).
- **`random_state`** (`int`): Seed for reproducibility in data splitting and some models.
- **`data_bunch`**: The original Scikit-Learn `Bunch` object returned by `load_breast_cancer`.
- **`X`** (`pandas.DataFrame`): Feature matrix.
- **`y`** (`pandas.Series`): Target labels.
- **`feature_names`** (`array-like`): Names of the input features.
- **`target_names`** (`array-like`): Names of the target classes.
- **`X_train`, `X_test`, `y_train`, `y_test`**: Training and testing subsets.
- **`models`** (`dict`): Mapping from model name to instantiated classifier (or pipeline).
- **`results`** (`dict`): Mapping from model name to computed metrics and reports.

**Methods:**

- **`load_data()`**  
  Loads the breast cancer dataset using `load_breast_cancer`, converts features to a `pandas.DataFrame` and labels to a `pandas.Series`. This makes it easier to inspect and manipulate the data.

- **`train_test_split()`**  
  Uses `sklearn.model_selection.train_test_split` to split `X` and `y` into training and testing sets. The method uses `stratify=y` to maintain class proportions and `random_state` for reproducibility.

- **`build_models()`**  
  Constructs three classification models with some parameter tuning:
  - **Logistic Regression** (`LogisticRegression`) wrapped in a `Pipeline` with `StandardScaler`. Uses `penalty='l2'`, `C=1.0`, `solver='lbfgs'`, and `max_iter=500`.
  - **Random Forest** (`RandomForestClassifier`) with `n_estimators=300`, `max_features='sqrt'`, and `criterion='gini'`. These parameters increase model capacity and diversity of trees.
  - **SVM with RBF kernel** (`SVC`) wrapped in a `Pipeline` with `StandardScaler`. Uses `kernel='rbf'`, `C=5.0`, `gamma='scale'`, and `probability=True`. The choice of RBF kernel and tuned `C` provides non-linear decision boundaries and strong performance.

- **`fit_models()`**  
  Iterates over all models in `self.models` and calls `.fit(X_train, y_train)` to train them.

- **`evaluate_models()`**  
  For each model, generates predictions on `X_test` and computes:
  - **Accuracy** (`accuracy_score`)
  - **Macro F1-score** (`f1_score(average='macro')`)
  - **Macro precision** (`precision_score(average='macro')`)
  - **Macro recall** (`recall_score(average='macro')`)
  - **Confusion matrix** (`confusion_matrix`)
  - **Classification report** (`classification_report`)
  
  These metrics are stored in `self.results[model_name]`.

- **`print_results()`**  
  Prints a formatted summary of metrics and the confusion matrix and classification report for each model.

- **`get_best_model_by_metric(metric='accuracy')`**  
  Compares models based on a chosen metric (default: accuracy) and returns the name and metrics of the best-performing model.

---

## Models and metrics

### Models implemented

1. **Logistic Regression**
   - Linear classifier with L2 regularization.
   - Benefits from standardized features.
   - Interpretable coefficients (odds ratios) and relatively fast training.

2. **Random Forest**
   - Ensemble of decision trees using bagging and feature subsampling.
   - Captures non-linear relationships and interactions.
   - Provides feature importance estimates, which can be useful for clinical interpretation.

3. **Support Vector Machine (RBF kernel)**
   - Maximizes the margin between classes in a transformed feature space.
   - RBF kernel allows flexible non-linear decision boundaries.
   - Tuned `C` and `gamma` for improved performance.

### Evaluation metrics

For each model, the following metrics are computed on the test set:

- **Accuracy:** Overall proportion of correctly classified samples.
- **Macro F1-score:** Harmonic mean of precision and recall, averaged across classes.
- **Macro precision:** Average precision across classes, treating each class equally.
- **Macro recall:** Average recall across classes, treating each class equally.
- **Confusion matrix:** Counts of true positives, true negatives, false positives, and false negatives per class.
- **Classification report:** Detailed per-class precision, recall, and F1.

These metrics allow comparison not only on overall correctness (accuracy) but also on how well each model balances false positives and false negatives—critical in a medical context.

---

## Limitations

- **Single dataset:** The models are trained and evaluated on a single dataset; generalization to other populations or imaging modalities is not guaranteed.
- **Simple tuning:** Parameter choices are manually tuned rather than exhaustively optimized via grid search or cross-validation. More systematic hyperparameter optimization could further improve performance.
- **Class imbalance:** While the dataset is not extremely imbalanced, more advanced techniques (e.g., class weighting, resampling) could be explored to further protect against bias.
- **Interpretability vs. performance:** Models like SVM and Random Forest may offer higher performance but are less interpretable than Logistic Regression, which can be a limitation in clinical decision-making.

---

## How to run

1. Create and activate a conda environment (optional but recommended):

   ```bash
   conda create -n scikit_env python=3.11
   conda activate scikit_env
   conda install scikit-learn pandas numpy scipy
