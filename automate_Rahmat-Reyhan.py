import os
import tempfile
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore")


# ============================================================
# 1. KONFIGURASI
# ============================================================

DATASET_URL = (
    "https://raw.githubusercontent.com/Nas-virat/"
    "Telco-Customer-Churn/main/Telco-Customer-Churn.csv"
)

EXPERIMENT_NAME = "Telco Customer Churn Classification"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 70)
print("TELCO CUSTOMER CHURN - AUTOMATED ML EXPERIMENT")
print("=" * 70)

print("\n[1] Loading dataset...")

df = pd.read_csv(DATASET_URL)

print(f"Dataset shape: {df.shape}")
print(f"Jumlah data: {len(df)}")


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

print("\n[2] Preprocessing dataset...")

data = df.copy()

# Hapus duplikasi
duplicate_count = data.duplicated().sum()

if duplicate_count > 0:
    print(f"Duplicate ditemukan: {duplicate_count}")
    data = data.drop_duplicates()
else:
    print("Tidak terdapat duplicate data.")

# Konversi TotalCharges menjadi numeric
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

# Konversi target Churn menjadi 0 dan 1
data["Churn"] = data["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Pisahkan fitur dan target
X = data.drop(columns=["Churn", "customerID"])
y = data["Churn"]

numeric_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    column
    for column in X.columns
    if column not in numeric_features
]

# Pipeline numerik
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Pipeline kategorikal
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training data : {len(X_train)}")
print(f"Testing data  : {len(X_test)}")


# ============================================================
# 5. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Melakukan evaluasi model menggunakan beberapa metrik.
    """

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_probability = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_probability)
    else:
        roc_auc = 0.0

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
        "roc_auc": roc_auc
    }

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    return metrics, report


# ============================================================
# 6. MODEL EXPERIMENTS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )
}


# ============================================================
# 7. MLFLOW SETUP
# ============================================================

print("\n[3] Setting up MLflow...")

# Mendukung MLflow tracking URI dari environment variable.
# Jika tidak tersedia, MLflow menggunakan local tracking.
tracking_uri = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///mlflow.db"
)

mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment(EXPERIMENT_NAME)

print(f"MLflow Tracking URI: {tracking_uri}")
print(f"Experiment: {EXPERIMENT_NAME}")


# ============================================================
# 8. TRAINING DAN MANUAL LOGGING
# ============================================================

results = []

print("\n[4] Training baseline models...")

for model_name, model in models.items():

    print(f"\nTraining: {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    metrics, report = evaluate_model(
        pipeline,
        X_test,
        y_test
    )

    results.append({
        "model": model_name,
        **metrics
    })

    # --------------------------------------------------------
    # MLflow manual logging
    # --------------------------------------------------------

    with mlflow.start_run(run_name=model_name):

        mlflow.log_param(
            "model_name",
            model_name
        )

        mlflow.log_param(
            "test_size",
            TEST_SIZE
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE
        )

        mlflow.log_param(
            "train_samples",
            len(X_train)
        )

        mlflow.log_param(
            "test_samples",
            len(X_test)
        )

        mlflow.log_param(
            "numeric_features",
            len(numeric_features)
        )

        mlflow.log_param(
            "categorical_features",
            len(categorical_features)
        )

        # Log model parameters
        model_params = model.get_params()

        for parameter, value in model_params.items():

            # MLflow parameter value harus string sederhana
            try:
                mlflow.log_param(
                    f"model_{parameter}",
                    value
                )
            except Exception:
                pass

        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(
                metric_name,
                metric_value
            )

        # Classification report sebagai artifact
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8"
        ) as report_file:

            report_file.write(report)
            report_path = report_file.name

        mlflow.log_artifact(
            report_path,
            artifact_path="evaluation"
        )

        os.remove(report_path)

        # Log trained model
        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            serialization_format="cloudpickle"
        )

        print(
            f"  Accuracy : {metrics['accuracy']:.4f}"
        )
        print(
            f"  Precision: {metrics['precision']:.4f}"
        )
        print(
            f"  Recall   : {metrics['recall']:.4f}"
        )
        print(
            f"  F1 Score : {metrics['f1']:.4f}"
        )
        print(
            f"  ROC-AUC  : {metrics['roc_auc']:.4f}"
        )


# ============================================================
# 9. BASELINE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("BASELINE MODEL RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))


# ============================================================
# 10. HYPERPARAMETER TUNING
# ============================================================

print("\n[5] Hyperparameter tuning Logistic Regression...")

logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ]
)

param_grid = {
    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],
    "model__solver": [
        "liblinear",
        "lbfgs"
    ]
}

grid_search = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

best_model = grid_search.best_estimator_

tuned_metrics, tuned_report = evaluate_model(
    best_model,
    X_test,
    y_test
)

print("\nBest parameters:")
print(grid_search.best_params_)

print(
    f"Best CV F1: "
    f"{grid_search.best_score_:.4f}"
)

print(
    f"Test F1: "
    f"{tuned_metrics['f1']:.4f}"
)


# ============================================================
# 11. LOG TUNED MODEL KE MLFLOW
# ============================================================

with mlflow.start_run(
    run_name="Logistic Regression Tuned"
):

    mlflow.log_param(
        "model_name",
        "Logistic Regression Tuned"
    )

    mlflow.log_param(
        "test_size",
        TEST_SIZE
    )

    mlflow.log_param(
        "random_state",
        RANDOM_STATE
    )

    mlflow.log_param(
        "cv",
        5
    )

    mlflow.log_param(
        "scoring",
        "f1"
    )

    mlflow.log_param(
        "best_C",
        grid_search.best_params_["model__C"]
    )

    mlflow.log_param(
        "best_solver",
        grid_search.best_params_["model__solver"]
    )

    mlflow.log_metric(
        "cv_best_f1",
        grid_search.best_score_
    )

    for metric_name, metric_value in tuned_metrics.items():
        mlflow.log_metric(
            metric_name,
            metric_value
        )

    # Classification report
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".txt",
        delete=False,
        encoding="utf-8"
    ) as report_file:

        report_file.write(tuned_report)
        tuned_report_path = report_file.name

    mlflow.log_artifact(
        tuned_report_path,
        artifact_path="evaluation"
    )

    os.remove(tuned_report_path)

    # Log tuned model
    mlflow.sklearn.log_model(
        best_model,
        artifact_path="model",
        serialization_format="cloudpickle"
    )


# ============================================================
# 12. TAMBAHKAN HASIL TUNING
# ============================================================

tuned_result = {
    "model": "Logistic Regression Tuned",
    **tuned_metrics
}

results_df = pd.concat(
    [
        results_df,
        pd.DataFrame([tuned_result])
    ],
    ignore_index=True
)

results_df = results_df.sort_values(
    by="f1",
    ascending=False
).reset_index(drop=True)


# ============================================================
# 13. SIMPAN HASIL EKSPERIMEN
# ============================================================

output_file = "model_comparison_results.csv"

results_df.to_csv(
    output_file,
    index=False
)

print("\n[6] Hasil eksperimen disimpan:")
print(output_file)


# ============================================================
# 14. LOG HASIL PERBANDINGAN SEBAGAI ARTIFACT
# ============================================================

with mlflow.start_run(
    run_name="Model Comparison Artifacts"
):

    mlflow.log_param(
        "experiment_purpose",
        "Model comparison and evaluation"
    )

    mlflow.log_param(
        "selection_metric",
        "F1 Score"
    )

    best_row = results_df.iloc[0]

    mlflow.log_param(
        "best_model_by_f1",
        best_row["model"]
    )

    mlflow.log_metric(
        "best_f1",
        best_row["f1"]
    )

    mlflow.log_artifact(
        output_file,
        artifact_path="results"
    )


# ============================================================
# 15. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("AUTOMATION SELESAI")
print("=" * 70)

print("\nHasil akhir:")
print(results_df.to_string(index=False))

print("\nBest model berdasarkan F1 Score:")
print(best_row["model"])

print(
    f"Best F1 Score: "
    f"{best_row['f1']:.4f}"
)

print("\nMLflow logging berhasil dilakukan.")