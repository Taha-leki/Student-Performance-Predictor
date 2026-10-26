
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, sep=";")

FEATURES = [
    "age",
    "studytime",
    "failures",
    "absences",
    "schoolsup",
    "paid",
    "activities",
    "higher",
    "traveltime",
    "goout",
]

X = df[FEATURES]
y = df["G3"]

numeric_features = X.select_dtypes(include="number").columns.tolist()
categorical_features = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=5, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, random_state=42
    ),
}

results = []
best_pipeline = None
best_name = None
best_mae = float("inf")

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 3),
    })

    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

    if mae < best_mae:
        best_mae = mae
        best_name = name
        best_pipeline = pipeline

joblib.dump(best_pipeline, MODELS_DIR / "best_model.joblib")

results_df = pd.DataFrame(results).sort_values("MAE")
results_df.to_csv(BASE_DIR / "model_results.csv", index=False)

print("\nUpdated model comparison:")
print(results_df.to_string(index=False))
print(f"\nBest model: {best_name}")
print("Updated model saved successfully.")
