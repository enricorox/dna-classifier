import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from xgboost import plot_importance

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
print(f"\tData points: {len(X_train)}")
print(f"\t\tlabel(0) counts: {(y_train['label'] == 0).sum()}")
print(f"\t\tlabel(1) counts: {(y_train['label'] == 1).sum()}")

print("Loading test data...")
test_data = pd.read_csv(test_data_file)
X_test, y_test = test_data.drop('label', axis=1), test_data[['label']]
print(f"\tData points: {len(X_test)}")
print(f"\t\tlabel(0) counts: {(y_test['label'] == 0).sum()}")
print(f"\t\tlabel(1) counts: {(y_test['label'] == 1).sum()}")

print("Loading validation data...")
validation_data = pd.read_csv(validation_data_file)
X_validation, y_validation = validation_data.drop('label', axis=1), validation_data[['label']]
print(f"\tData points: {len(X_validation)}")
print(f"\t\tlabel(0) counts: {(y_validation['label'] == 0).sum()}")
print(f"\t\tlabel(1) counts: {(y_validation['label'] == 1).sum()}")

# === use xgboost matrices ===
print("Transforming into DMatrices...")
dtrain = xgb.DMatrix(X_train, y_train)

"""
Test feature weights

Weight for each feature, defines the probability of each feature being selected when colsample_by* is being used.
All values must be greater than (or equal to) 0, otherwise a ValueError is thrown.
"""


def equal_weight():
    return [1 for _ in range(dtrain.num_col())]


def random_weights():
    return np.uint32([random.choice([0, 1]) for _ in range(dtrain.num_col())])


# great weight set to zero
def inverse_weight():
    fw = [1 for _ in range(dtrain.num_col())]
    zeros = [1, 43, 17, 50, 45, 25, 37, 44, 38, 63]
    for i in zeros:
        fw[i - 1] = 0
    return fw


def select_weight():
    fw = [0 for _ in range(dtrain.num_col())]
    ones = [1, 17, 25, 37, 38, 43, 44, 45, 50, 63]
    for i in ones:
        fw[i - 1] = 1
    return fw


#fw = equal_weight()
#fw = inverse_weight()
fw = select_weight()
dtrain.set_info(feature_weights=fw)
col_sample_bynode = .1

dvalidation = xgb.DMatrix(X_validation, y_validation)
dtest = xgb.DMatrix(X_test, y_test)
# ========================================

# Define hyperparameters
# params = {"objective": "reg:squarederror", "tree_method": "gpu_hist"}
# params = {"objective": "reg:squarederror", "tree_method": "hist", "colsample_bynode": 0.75}
# params = {"objective": "binary:hinge", "tree_method": "hist"}
params = {"verbosity": 2, "device": "cpu", "objective": "binary:hinge", "tree_method": "hist",
          "colsample_bytree": col_sample_bynode, "max_depth": 12}
params["interaction_constraints"] = [["kmer43", "kmer25", "kmer17"]]
# train and validation
evals = [(dtrain, "training"), (dvalidation, "validation")]
print("Training...")
num_boost_round = 10
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=evals,
    verbose_eval=10,
    early_stopping_rounds=50
)

print("Prediction...")
preds = model.predict(dtest)

rmse = mean_squared_error(y_test, preds, squared=False)
print(f"RMSE of the base model: {rmse:.5f}")

accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy * 100:.2f}%")

print("Confusion matrix:")
conf_m = confusion_matrix(y_test, preds)
print(conf_m)

tp = conf_m[0][0]
fn = conf_m[0][1]
fp = conf_m[1][0]
prec = tp / (tp + fp)
rec = tp / (tp + fn)
f1 = 2 * prec * rec / (prec + rec)
print(f"Precision = {prec}")
print(f"Recall = {rec}")
print(f"F1 = {f1}")

print("Features importance:")
plot_importance(model, height=2)
plt.savefig("feature-importance.png")  # too small

# ‘weight’: the number of times a feature is used to split the data across all trees.
scores = model.get_score(importance_type="weight")
print(scores)
print("Done!")

graph = xgb.to_graphviz(model, num_trees=0)
graph.render(filename=f"decision-tree-0.gv{f1}")
