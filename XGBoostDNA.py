import graphviz
import pandas as pd
import xgboost as xgb
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from xgboost import Booster


class XGBoostDNA:
    bst: Booster
    num_trees: int
    best_it: int
    best_score: float

    dtrain: xgb.DMatrix
    dvalidation: xgb.DMatrix
    dtest: xgb.DMatrix

    y_pred: ndarray
    y_test: ndarray

    def __init__(self, model_name="xgbtree", num_trees=10):
        self.model_name = model_name
        self.num_trees = num_trees

        print(f"Using XGBoost version {xgb.__version__}")

    def read_datasets(self, train_data_file, test_data_file, validation_data_file=None, feature_weights=None):
        print("Loading training data...")
        train_data = pd.read_csv(train_data_file)
        X_train, y_train = train_data.drop('label', axis=1), train_data[['label']]
        print(f"\tData points: {len(X_train)}")
        print(f"\t\tlabel(0) counts: {(y_train['label'] == 0).sum()}")
        print(f"\t\tlabel(1) counts: {(y_train['label'] == 1).sum()}")
        print("Transforming into DMatrices...")
        self.dtrain = xgb.DMatrix(X_train, y_train)
        del train_data

        print("Loading test data...")
        test_data = pd.read_csv(test_data_file)
        X_test, y_test = test_data.drop('label', axis=1), test_data[['label']]
        print(f"\tData points: {len(X_test)}")
        print(f"\t\tlabel(0) counts: {(y_test['label'] == 0).sum()}")
        print(f"\t\tlabel(1) counts: {(y_test['label'] == 1).sum()}")
        print("Transforming into DMatrices...")
        self.y_test = y_test
        self.dtest = xgb.DMatrix(X_test)
        del test_data

        if validation_data_file is not None:
            print("Loading validation data...")
            validation_data = pd.read_csv(validation_data_file)
            X_validation, y_validation = validation_data.drop('label', axis=1), validation_data[['label']]
            print(f"\tData points: {len(X_validation)}")
            print(f"\t\tlabel(0) counts: {(y_validation['label'] == 0).sum()}")
            print(f"\t\tlabel(1) counts: {(y_validation['label'] == 1).sum()}")
            print("Transforming into DMatrices...")
            self.dvalidation = xgb.DMatrix(X_validation, y_validation)
            del validation_data

        if feature_weights is not None:
            assert len(feature_weights) == self.dtrain.num_col()
            self.dtrain.set_info(feature_weights=feature_weights)

    def fit(self, params=None, evals=None):
        if self.dtrain is None:
            raise Exception("Need to load training datasets first!")

        if params is None:
            params = {"verbosity": 1, "device": "cuda", "objective": "binary:hinge", "tree_method": "hist",
                      "colsample_bytree": .8}

        if evals is None:
            if self.dvalidation is None:
                evals = [(self.dtrain, "training")]
            else:
                evals = [(self.dtrain, "training"), (self.dvalidation, "validation")]

        self.bst = xgb.train(params=params, dtrain=self.dtrain,
                             num_boost_round=self.num_trees,
                             evals=evals,
                             verbose_eval=10,
                             early_stopping_rounds=50
                             )

        # update number of trees in case of early stopping
        self.num_trees = self.bst.num_boosted_rounds()
        self.best_it = self.bst.best_iteration
        self.best_score = self.bst.best_score

    def predict(self, iteration_range=None):
        if iteration_range is None:
            iteration_range = (0, self.best_it)

        self.y_pred = self.bst.predict(self.dtest, iteration_range=iteration_range)

    def print_stats(self):
        print("\nPrediction stats:")

        print(f"Best score: {self.best_score}")
        print(f"Best iteration: {self.best_it}")

        conf_mat = confusion_matrix(self.y_test, self.y_pred)
        true_neg = conf_mat[0][0]
        true_pos = conf_mat[1][1]
        false_neg = conf_mat[1][0]
        false_pos = conf_mat[0][1]

        assert (true_pos + false_neg) == sum(self.y_test["label"])
        assert (true_neg + false_pos) == len(self.y_test["label"]) - sum(self.y_test["label"])
        assert (true_neg + true_pos + false_neg + false_pos) == len(self.y_test["label"])

        print(f"TN={true_neg}\tFP={false_pos}")
        print(f"FN={false_neg}\tTP={true_pos}")

        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        print(f"Accuracy = {accuracy}")

    def plot_trees(self, tree_set=None):
        if tree_set is None:
            tree_set = range(self.num_trees)

        for i in tree_set:
            graph: graphviz.Source
            graph = xgb.to_graphviz(self.bst, num_trees=i)
            graph.render(filename=f"{self.model_name}-{i}", directory="trees", format="png", cleanup=True)


if __name__ == "__main__":
    dataset_folder = "../dna-embeddings/small/"
    train_data_file = dataset_folder + "training-data.csv"
    test_data_file = dataset_folder + "test-data.csv"
    validation_data_file = dataset_folder + "validation-data.csv"

    clf = XGBoostDNA(num_trees=100)
    clf.read_datasets(train_data_file, test_data_file, validation_data_file=validation_data_file)
    clf.fit()
    clf.predict()
    clf.print_stats()
    clf.plot_trees()
