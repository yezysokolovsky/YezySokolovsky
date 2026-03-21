import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

data = fetch_california_housing()
X, y = data.data, data.target
feature_names = data.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

models = {
    "Gradient Boosting": GradientBoostingRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Extra Trees": ExtraTreesRegressor(),
    "XGBoost": XGBRegressor(verbosity=0),
    "LightGBM": LGBMRegressor(),
    "Linear Regression": LinearRegression()
}

results = []

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results.append([name, mae, rmse, r2, cv_scores.mean()])

results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2", "CV_R2"]
).sort_values(by="R2", ascending=False)

print(results_df)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring="r2", n_jobs=-1)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

preds = best_rf.predict(X_test)

importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure()
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=45)
plt.tight_layout()
plt.savefig("feature_importance.png")

plt.figure()
plt.scatter(y_test, preds)
plt.xlabel("Real")
plt.ylabel("Predicted")
plt.savefig("predictions.png")
