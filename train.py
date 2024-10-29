# train.py
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# Generate synthetic dataset with 1 million samples and 10 features
X, y = make_regression(n_samples=100000, n_features=10, noise=0.1, random_state=42)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("model_comparison")

# Train and log Linear Regression model
with mlflow.start_run(run_name="Linear_Regression"):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_mse = mean_squared_error(y_test, lr_preds)
    mlflow.log_metric("mse", lr_mse)
    mlflow.sklearn.log_model(lr, "linear_regression_model")
    print(f"Linear Regression MSE: {lr_mse}")

# Train and log Random Forest model
with mlflow.start_run(run_name="Random_Forest"):
    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mse = mean_squared_error(y_test, rf_preds)
    mlflow.log_metric("mse", rf_mse)
    mlflow.sklearn.log_model(rf, "random_forest_model")
    print(f"Random Forest MSE: {rf_mse}")
