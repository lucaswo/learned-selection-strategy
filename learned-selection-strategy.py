from __future__ import annotations
import argparse

import os
import time
from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor


def smape(y_true: np.array, y_pred: np.array) -> np.array:
    return 200*np.mean(np.abs(y_pred-y_true)/(np.abs(y_true) + np.abs(y_pred)))

def sape(y_true: np.array, y_pred: np.array) -> np.array:
    return 200*(np.abs(y_pred-y_true)/(np.abs(y_true) + np.abs(y_pred)))

def mse(y_true: np.array, y_pred: np.array) -> np.array:
    return (y_true - y_pred)**2

def mae(y_true: np.array, y_pred: np.array) -> np.array:
    return np.abs(y_true - y_pred)

def rel_err(y_true: np.array, y_pred: np.array) -> np.array:
    return (y_pred - y_true).abs() / y_true

def prepare_data(raw_data: pd.DataFrame, bw_cols: list[str]) -> pd.DataFrame:
    data = raw_data.groupby(["settingIdx", "format", "settingGroup"], as_index=False).mean()

    data.loc[:,bw_cols] = data.loc[:,bw_cols].div(data["countValuesSmall"], axis=0)
    
    data["compressed size [byte]"] = data["compressed size [byte]"]/data["countValuesSmall"]*8

    data.insert(13, "minBucket", 
                data.loc[:,bw_cols].ne(0).idxmax(axis=1).str.replace("bwHist_", "").astype(float).values)
    data.insert(14, "maxBucket", data["bitwidth"].values)
    data.insert(15, "Avg", data[bw_cols].dot(range(1,65)).values)
    data.insert(16, "numBuckets", data.loc[:,bw_cols].ne(0).sum(axis=1).astype(float).values)
    data.insert(17, "Std", data[bw_cols].std(axis=1).values)
    data.insert(18, "Skew", data[bw_cols].skew(axis=1).values)
    data.insert(19, "Kurt", data[bw_cols].kurtosis(axis=1).values)
    
    offset = (data.columns == bw_cols[0]).argmax() - 1
    data["Min"] = data.values[range(len(data)),data["minBucket"].astype(int)+offset]
    data["Max"] = data.values[range(len(data)),data["maxBucket"].astype(int)+offset]
    
    return data

def train_model(data_train: pd.DataFrame, data_test: pd.DataFrame, algorithm: str, objective: str, features: list[int], est_range: list[int], d_range: list[int]) -> tuple[GradientBoostingRegressor, pd.DataFrame, pd.DataFrame]:
    best_err = np.inf
    timings = []

    for est in est_range:
        for d in d_range: 
            model = GradientBoostingRegressor(n_estimators = est, max_depth = d, random_state = 1337)
            obj_idx = data_train.columns.get_loc(objective)

            train_time = time.time()
            model.fit(data_train.iloc[:,features].values, data_train.iloc[:,obj_idx])
            train_time = time.time() - train_time

            pred_time = time.time()
            y_pred = model.predict(data_test.iloc[:,features].values)
            pred_time = time.time() - pred_time

            y_test = data_test.iloc[:,obj_idx]

            timings.append((est, d, train_time, pred_time))

            eor = pd.DataFrame({"pred": y_pred, 
                                "mse": mse(y_test, y_pred).values,
                                "mae": mae(y_test, y_pred).values,
                                "sape": sape(y_test, y_pred).values})
            print("""Trained model {}x{} for {} ({}) in {:.4f}s. Predicted in {:.4f}s ({:.4f}ms per sample)
                Errors: MSE={:.4f}, MAE={:.4f}, SMAPE={:.2f}%""". \
                format(est, d, algorithm, objective, train_time, pred_time, 
                        pred_time/len(data_test)*1000, eor["mse"].mean(), eor["mae"].mean(), 
                        eor["sape"].mean()))

            if eor["sape"].mean() < best_err:
                best_model = model
                best_err = eor["sape"].mean()
                errors = pd.concat([data_test.iloc[:,features].reset_index(drop=True), 
                                    y_test.reset_index(drop=True), eor], axis=1)

    timings = pd.DataFrame(timings, columns=['trees', 'depth', 'train time', 'forward pass'])
    return best_model, errors, timings

def use_selection_strategy(pool: defaultdict, data_test: pd.DataFrame, objective: str, features: list[int]) -> pd.DataFrame:
    data_pred = data_test[["settingIdx", "format", objective]].copy()
    for p_algo in data_test["format"].unique():
        preds = pool[p_algo][objective].predict(data_test[data_test["format"] == p_algo].iloc[:,features].values)
        data_pred.loc[data_pred["format"] == p_algo, ["pred"]] = preds

    algo_idx = data_pred.groupby("settingIdx", as_index=False).agg({"pred": "idxmin", objective: "idxmin"})
    
    find_best = pd.concat((data_pred.loc[algo_idx[objective]][["format", objective]].reset_index(drop=True),
                    data_pred.loc[algo_idx["pred"]][["format", objective]].reset_index(drop=True)),
                    axis=1)
    find_best.columns = ['correct algorithm', 'true', 'predicted algorithm', 'pred']
    find_best["equal"] = (find_best["correct algorithm"] == find_best["predicted algorithm"]).astype(int)
    find_best["sape"] = sape(find_best["true"], find_best["pred"])
    
    find_best.insert(0, "objective", objective)

    print("Objective: {}, accuracy {:.2f}%".format(objective, find_best["equal"].sum()/len(find_best)*100))
    print("Objective: {}, slowdown {:.2f}%".format(objective, find_best.loc[find_best["equal"] == 0, "sape"].mean()))
    
    return find_best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args(argv)

    #print(args)

    data_train = {}
    data_test = {}

    # discover all algorithms
    # measurement data contains the features, objectives and bit width histograms
    for f_name in os.listdir("measurements/"):
        data = pd.read_csv("measurements/{}".format(f_name), sep="\t")
        if "train" in f_name:
            print("Found {}.".format(f_name))
            data_train[data["format"].iloc[0]] = data
        elif "test" in f_name:
            data_test[data["format"].iloc[0]] = data
    
    # the model pool
    models = defaultdict(dict)
    errors = defaultdict(dict)
    timings = defaultdict(dict)

    obj_columns = ["runtime compr ram2ram [µs]", 
                   "runtime decompr ram2ram [µs]", 
                   "runtime decompr ram2reg [µs]", 
                   "runtime compr cache2ram [µs]",
                   "compressed size [byte]"]
    features = [21,22,23,13,14,15,16,17,18,19]

    # train model for each algorithm
    for algo in data_train.keys():
        assert algo in data_test.keys(), "{algo} not in test data"
        # train model for each algorithm x objective
        for objective in obj_columns:
            models[algo][objective], errors[algo][objective], timings[algo][objective] = train_model(data_train[algo], data_test[algo], algo, objective, features, range(10,60,10), range(3,7))

    # test predictions (accuracy and slowdown)
    # need full data to find best algorithm for each settingIdx
    data_test_full = pd.concat(data_test.values(), ignore_index = True)
    for objective in obj_columns:
        use_selection_strategy(models, data_test_full, objective, features)
    
    return 0

if __name__ == "__main__":
    main()