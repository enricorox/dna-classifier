import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier, Booster


class XGBoostDNA:
    def __init__(self, num_trees=10):
        self.bst = None

        self.dtrain = None
        self.dvalidation = None
        self.dtest = None

        self.num_trees = num_trees

        print(f"Using XGBoost version {xgb.__version__}")

    def read_datasets(self, train_data_file, test_data_file, validation_data_file=None, feature_weights=None):
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

        print(validation_data_file)
        if validation_data_file is not None:
            print("Loading validation data...")
            validation_data = pd.read_csv(validation_data_file)
            X_validation, y_validation = validation_data.drop('label', axis=1), validation_data[['label']]
            print(f"\tData points: {len(X_validation)}")
            print(f"\t\tlabel(0) counts: {(y_validation['label'] == 0).sum()}")
            print(f"\t\tlabel(1) counts: {(y_validation['label'] == 1).sum()}")

            self.dvalidation = xgb.DMatrix(X_validation, y_validation)

        # === use xgboost matrices ===
        print("Transforming into DMatrices...")
        self.dtrain = xgb.DMatrix(X_train, y_train)
        self.dtest = xgb.DMatrix(X_test, y_test)

        if feature_weights is not None:
            assert len(feature_weights) == self.dtrain.num_col()
            self.dtrain.set_info(feature_weights=feature_weights)

    def fit(self, params=None, evals=None):
        if self.dtrain is None:
            raise Exception("Need to load training datasets first!")

        if self.dvalidation is None:
            raise Exception("Need to load validation datasets first!")

        if params is None:
            params = {"verbosity": 1, "device": "cuda", "objective": "binary:hinge", "tree_method": "hist",
                      "colsample_bytree": .8}

        if evals is None:
            evals = [(self.dtrain, "training"), (self.dvalidation, "validation")]

        self.bst = xgb.train(params=params, dtrain=self.dtrain,
                             num_boost_round=self.num_trees,
                             evals=evals,
                             verbose_eval=10,
                             early_stopping_rounds=50
                             )

        # update in case of early stopping
        self.num_trees = self.bst.num_boosted_rounds()

    def predict(self, iteration_range=None):
        if iteration_range is None:
            iteration_range = (0, self.num_trees)

        self.bst.predict(self.dtest, iteration_range=iteration_range)


if __name__ == "__main__":
    dataset_folder = "../dna-embeddings/small/"
    train_data_file = dataset_folder + "training-data.csv"
    test_data_file = dataset_folder + "test-data.csv"
    validation_data_file = dataset_folder + "validation-data.csv"

    clf = XGBoostDNA(num_trees=100)
    clf.read_datasets(train_data_file, test_data_file, validation_data_file=validation_data_file)
    clf.fit()
    clf.predict()
