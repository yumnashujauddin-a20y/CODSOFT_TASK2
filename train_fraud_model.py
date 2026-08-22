import os
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = r"C:\Users\yousr\OneDrive\Desktop\codesoft\fraudTest.csv"

OUTPUT_DIR = r"C:\Users\yousr\OneDrive\Desktop\codesoft\fraud_model_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "is_fraud"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found!")

print("\nTarget:", TARGET)

print("\nFraud distribution:")
print(df[TARGET].value_counts())

print("\nFraud percentage:")
print(df[TARGET].value_counts(normalize=True) * 100)


# ============================================================
# 4. CREATE FEATURES
# ============================================================

X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()


# ------------------------------------------------------------
# Convert transaction date into useful features
# ------------------------------------------------------------

if "trans_date_trans_time" in X.columns:

    X["trans_date_trans_time"] = pd.to_datetime(
        X["trans_date_trans_time"],
        errors="coerce"
    )

    X["transaction_hour"] = X["trans_date_trans_time"].dt.hour
    X["transaction_day"] = X["trans_date_trans_time"].dt.day
    X["transaction_month"] = X["trans_date_trans_time"].dt.month
    X["transaction_dayofweek"] = X["trans_date_trans_time"].dt.dayofweek

    # Original datetime is removed after extracting useful information
    X.drop(columns=["trans_date_trans_time"], inplace=True)


# ------------------------------------------------------------
# Convert DOB into age
# ------------------------------------------------------------

if "dob" in X.columns:

    X["dob"] = pd.to_datetime(
        X["dob"],
        errors="coerce"
    )

    current_year = 2026

    X["age"] = current_year - X["dob"].dt.year

    X.drop(columns=["dob"], inplace=True)


# ============================================================
# 5. REMOVE PURE IDENTIFIERS
# ============================================================

columns_to_remove = [
    "Unnamed: 0",
    "trans_num",
    "unix_time"
]

columns_to_remove = [
    col for col in columns_to_remove
    if col in X.columns
]

X.drop(columns=columns_to_remove, inplace=True)

print("\nRemoved identifier columns:")
print(columns_to_remove)


# ============================================================
# 6. HANDLE CREDIT CARD NUMBER
# ============================================================

# cc_num is an identifier rather than a meaningful numerical
# measurement. Treat it as categorical if available.

if "cc_num" in X.columns:

    X["cc_num"] = X["cc_num"].astype(str)


# ============================================================
# 7. FEATURE INFORMATION
# ============================================================

numeric_columns = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

print("\nNumeric features:", len(numeric_columns))
print(numeric_columns)

print("\nCategorical features:", len(categorical_columns))
print(categorical_columns)


# ============================================================
# 8. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )
    )
])


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ]
)


# ============================================================
# 9. RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)


# ============================================================
# 10. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "model",
        model
    )
])


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 12. TRAIN
# ============================================================

print("\n====================================")
print("TRAINING MODEL")
print("====================================")

pipeline.fit(X_train, y_train)

print("\nTraining completed!")


# ============================================================
# 13. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = pipeline.predict(X_test)

y_probability = pipeline.predict_proba(X_test)[:, 1]


# ============================================================
# 14. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n====================================")
print("MODEL PERFORMANCE")
print("====================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n====================================")
print("CLASSIFICATION REPORT")
print("====================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Not Fraud",
            "Fraud"
        ],
        zero_division=0
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n====================================")
print("CONFUSION MATRIX")
print("====================================")

print(cm)

cm_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Not Fraud",
        "Fraud"
    ]
)

cm_display.plot()

plt.title("Fraud Detection - Confusion Matrix")
plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# 17. ROC CURVE
# ============================================================

RocCurveDisplay.from_predictions(
    y_test,
    y_probability
)

plt.title("Fraud Detection - ROC Curve")
plt.tight_layout()

roc_path = os.path.join(
    OUTPUT_DIR,
    "roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300
)

plt.show()

plt.close()


# ============================================================
# 18. SAVE MODEL
# ============================================================

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "fraud_model.joblib"
)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\n====================================")
print("MODEL SAVED")
print("====================================")

print(MODEL_PATH)


# ============================================================
# 19. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics_path = os.path.join(
    OUTPUT_DIR,
    "model_metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 20. SAVE CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Not Fraud",
        "Fraud"
    ],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.csv"
)

report_df.to_csv(
    report_path
)


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\n====================================")
print("ALL FILES SAVED")
print("====================================")

print("Model:")
print(MODEL_PATH)

print("\nMetrics:")
print(metrics_path)

print("\nClassification report:")
print(report_path)

print("\nConfusion matrix:")
print(cm_path)

print("\nROC curve:")
print(roc_path)

print("\nDONE!")