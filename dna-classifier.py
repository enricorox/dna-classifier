import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_importance
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score

import xgboost as xgb

warnings.filterwarnings("ignore")

print("Loading data...")
data = pd.read_csv("data.csv")

# Extract feature and target arrays
X, y = data.drop('label', axis=1), data[['label']]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)

# using dmatrices for better performances
print("Transforming into DMatrices...")
dtrain_reg = xgb.DMatrix(X_train, y_train)
dtest_reg = xgb.DMatrix(X_test, y_test)

# Define hyperparameters
params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}

print("Training...")
num_boost_round = 100
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=num_boost_round,
)

print("Prediction...")
preds = model.predict(dtest_reg)
preds_rounded = [int(round(y)) for y in preds]

# rmse = mean_squared_error(y_test, preds, squared=False)
# print(f"RMSE of the base model: {rmse:.5f}")
rmse = mean_squared_error(y_test, preds_rounded, squared=False)
print(f"RMSE of the base model with rounded numbers: {rmse:.5f}")

accuracy = accuracy_score(y_test, preds_rounded)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("Confusion matrix:")
conf_m = confusion_matrix(y_test, preds_rounded)
print(conf_m)

print("Plotting features importance..")
plot_importance(model)
plt.savefig("feature-importance.png")

print("Done!")
