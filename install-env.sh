# environment for eXtreme Gradient Boosting (XGBoost)

conda create -y -n xgb python==3.8
conda activate xgb

yes | pip install --yes xgboost==2.0.0 graphviz pandas matplotlib numpy scikit-learn

# Use CPU only
#conda install -c conda-forge py-xgboost-cpu

# Use NVIDIA GPU
# conda install -c conda-forge py-xgboost-gpu