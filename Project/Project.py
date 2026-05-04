# =========================================================
# USA HOUSING PRICE PREDICTION — FULL PROJECT (CLEAN)
# =========================================================
# ===== 1. IMPORTS =====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, BaggingRegressor,
    AdaBoostRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# =========================================================
# ===== 2. DATA LOADING =====
# =========================================================
df = pd.read_csv("USA_Housing.csv")
df.drop(columns=["Address"], inplace=True)

print("Dataset Shape:", df.shape)
print(df.head())

# =========================================================
# ===== 3. DATA ANALYSIS =====
# =========================================================

# Correlation Heatmap
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()

# Price Distribution
plt.figure()
plt.hist(df["Price"], bins=50)
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.figure()
plt.boxplot(df["Price"], vert=False)
plt.title("Price Boxplot")
plt.show()

# =========================================================
# ===== 4. SPLIT DATA =====
# =========================================================
X = df.drop("Price", axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# =========================================================
# ===== 5. MODELS =====
# =========================================================
models = {
    "Linear": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
    "Bagging": BaggingRegressor(random_state=42),
    "Pasting": BaggingRegressor(bootstrap=False, random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# =========================================================
# ===== 6. TRAINING & EVALUATION =====
# =========================================================
results = []
predictions = {}

for name, model in models.items():
    
    if name in ["SVR", "KNN"]:
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    
    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = mean_absolute_percentage_error(y_test, pred) * 100
    
    results.append([name, r2, rmse, mape])
    predictions[name] = pred

# Result Table
result_df = pd.DataFrame(results, columns=["Model", "R2", "RMSE", "MAPE"])
result_df = result_df.sort_values(by="R2", ascending=False)

print("\nModel Comparison:")
print(result_df)

# =========================================================
# ===== 7. MODEL COMPARISON GRAPHS =====
# =========================================================

# R2 Comparison
plt.figure()
plt.barh(result_df["Model"], result_df["R2"])
plt.title("Model Comparison - R2 Score")
plt.xlabel("R2 Score")
plt.gca().invert_yaxis()
plt.show()

# RMSE Comparison
plt.figure()
plt.barh(result_df["Model"], result_df["RMSE"])
plt.title("Model Comparison - RMSE")
plt.xlabel("RMSE")
plt.gca().invert_yaxis()
plt.show()

# MAPE Comparison
plt.figure()
plt.barh(result_df["Model"], result_df["MAPE"])
plt.title("Model Comparison - MAPE (%)")
plt.xlabel("MAPE")
plt.gca().invert_yaxis()
plt.show()

# =========================================================
# ===== 8. BEST MODEL ANALYSIS =====
# =========================================================
best_model_name = result_df.iloc[0]["Model"]
best_pred = predictions[best_model_name]

print("\nBest Model:", best_model_name)

# Actual vs Predicted
plt.figure()
plt.scatter(y_test, best_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted ({best_model_name})")

# Perfect line
min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.show()

# Residual Plot
residuals = y_test - best_pred

plt.figure()
plt.scatter(best_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# Residual Distribution
plt.figure()
plt.hist(residuals, bins=40)
plt.title("Residual Distribution")
plt.show()

# =========================================================
# ===== 9. FEATURE IMPORTANCE (if applicable) =====
# =========================================================
best_model = models[best_model_name]

if hasattr(best_model, "feature_importances_"):
    importance = pd.Series(best_model.feature_importances_, index=X.columns)
    importance = importance.sort_values()

    plt.figure()
    importance.plot(kind="barh")
    plt.title("Feature Importance")
    plt.show()

# =========================================================
# ===== 10. CONCLUSION =====
# =========================================================
print("\n===== FINAL SUMMARY =====")
print(result_df)

print("\nKey Insights:")
print("- Linear relationships are strong in dataset")
print("- Tree models may overfit if not tuned")
print("- Best model gives highest R2 and lowest RMSE")
