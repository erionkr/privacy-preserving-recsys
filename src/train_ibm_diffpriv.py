# src/train_ibm_diffpriv.py

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import IBM DiffPrivLib
from diffprivlib.models import LinearRegression as DPLinearRegression

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    # Normalize user/item to [0,1]
    max_user = df["user"].max()
    max_item = df["item"].max()
    df["user_norm"] = df["user"] / float(max_user)
    df["item_norm"] = df["item"] / float(max_item)
    X = df[["user_norm", "item_norm"]].values.astype(float)
    y = df["rating"].values.astype(float)
    return X, y

def main(args):
    # 1) Load and normalize data
    X_train, y_train = load_and_preprocess(args.train_csv)
    X_test,  y_test  = load_and_preprocess(args.test_csv)

    # 2) Define DP linear regression, now with bounds_X and bounds_y
    #    Feature bounds: user_norm ∈ [0,1], item_norm ∈ [0,1]
    #    Label bounds: rating ∈ [1,5]
    dp_lr = DPLinearRegression(
        epsilon=args.epsilon,
        bounds_X=((0.0, 1.0), (0.0, 1.0)),
        bounds_y=(1.0, 5.0),
        fit_intercept=True
    )

    # 3) Train the model
    dp_lr.fit(X_train, y_train)

    # 4) Make predictions + calculate metrics
    y_pred_train = dp_lr.predict(X_train)
    y_pred_test  = dp_lr.predict(X_test)

    train_mse = mean_squared_error(y_train, y_pred_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mse  = mean_squared_error(y_test,  y_pred_test)
    test_mae  = mean_absolute_error(y_test,  y_pred_test)

    print(f"[IBM-DP] ε = {args.epsilon:.2f}")
    print(f"[IBM-DP] Train MSE: {train_mse:.4f} | Train MAE: {train_mae:.4f}")
    print(f"[IBM-DP] Test  MSE: {test_mse: .4f} | Test  MAE: {test_mae:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DP-linear regression (IBM DiffPrivLib)")
    parser.add_argument("--train_csv", type=str, default="../Dataset/train.csv", help="Path to train CSV file")
    parser.add_argument("--test_csv",  type=str, default="../Dataset/test.csv",  help="Path to test CSV file")
    parser.add_argument("--epsilon",   type=float, default=1.0, help="Privacy budget (ε) for DP linear regression")
    args = parser.parse_args()
    main(args)
