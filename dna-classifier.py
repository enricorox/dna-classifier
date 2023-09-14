import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import plot_importance
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score

import xgboost as xgb

warnings.filterwarnings("ignore")

dataset_folder = "../dna-embeddings/small/"
train_data_file = dataset_folder + "training-data.csv"
test_data_file = dataset_folder + "test-data.csv"
validation_data_file = dataset_folder + "validation-data.csv"

print(f"Using XGBoost version {xgb.__version__}")

# === read data & extract feature and target arrays===
print("Loading training data...")
train_data = pd.read_csv(train_data_file)
X_train, y_train = train_data.drop('label', axis=1), train_data[['label']]
print("Loading test data...")
test_data = pd.read_csv(test_data_file)
X_test, y_test = test_data.drop('label', axis=1), test_data[['label']]
print("Loading validation data...")
validation_data = pd.read_csv(validation_data_file)
X_validation, y_validation = validation_data.drop('label', axis=1), validation_data[['label']]

# === use xgboost matrices ===
print("Transforming into DMatrices...")
dtrain_reg = xgb.DMatrix(X_train, y_train)

# test feature weights
fw = [1/(i+1) for i in range(dtrain_reg.num_col() - 1)]
#dtrain_reg.set_info(feature_weights=fw)

dtest_reg = xgb.DMatrix(X_test, y_test)
# ======

# Define hyperparameters
params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
# params = {"objective": "reg:squarederror", "tree_method": "hist"}

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

print("Features importance:")
# plot_importance(model)
# plt.savefig("feature-importance.png") # too small
model.get_score(importance_type="gain")

print("Done!")
