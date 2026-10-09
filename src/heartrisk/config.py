"""Central configuration: paths, schema and constants used across the project."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "heart_cleveland_raw.csv"
CLEAN_FILE = PROCESSED_DIR / "heart_clean.csv"
MODEL_FILE = MODEL_DIR / "model.joblib"
MODEL_META_FILE = MODEL_DIR / "model_meta.json"

# Primary source: UCI ML Repository (Cleveland subset, 303 rows, '?' for missing)
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
# Fallback mirror of the same 303 Cleveland records (ISLR "Heart.csv" layout)
MIRROR_URL = "https://raw.githubusercontent.com/rashida048/Datasets/master/Heart.csv"

# Column order of processed.cleveland.data (UCI documentation, attributes 3,4,9,...,58)
UCI_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]

TARGET = "target"

# Feature groups used by the preprocessing pipeline
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
ENGINEERED_FEATURES = ["hr_reserve_pct"]
INPUT_FEATURES = UCI_COLUMNS[:-1]  # the 13 raw attributes the API accepts

# Valid values / ranges (used for cleaning and API validation)
VALID_CODES = {
    "sex": {0, 1},
    "fbs": {0, 1},
    "exang": {0, 1},
    "cp": {1, 2, 3, 4},
    "restecg": {0, 1, 2},
    "slope": {1, 2, 3},
    "thal": {3, 6, 7},
}
VALID_RANGES = {
    "age": (18, 100),
    "trestbps": (70, 220),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0.0, 7.0),
    "ca": (0, 3),
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
EXPERIMENT_NAME = "heart-disease-risk"
