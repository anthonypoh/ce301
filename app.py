import os
from flask import Flask, jsonify
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from services import download, unzip, correlation, predict

app = Flask(__name__)

LTA_COE_BIDDING_RESULT = "https://datamall.lta.gov.sg/content/dam/datamall/datasets/Facts_Figures/Vehicle%20Registration/COE%20Bidding%20Results.zip"
directory = os.getcwd()


@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify(message="Hello from Python microservice!")


@app.route("/api/get-prediction/<int:quota>/<string:cat>", methods=["GET"])
def getPrediction(quota, cat):
    zip_file_path = directory + "/data/file.zip"
    csv_file_pattern = "*-coe_results.csv"

    download(LTA_COE_BIDDING_RESULT, "/data")
    df = unzip(zip_file_path, csv_file_pattern)

    prediction = predict(df, quota, cat)

    return jsonify(quota=quota, category=cat, prediction=prediction)


@app.route("/api/get-correlation", methods=["GET"])
def getCorrelation():
    zip_file_path = directory + "/data/file.zip"
    csv_file_pattern = "*-coe_results.csv"

    download(LTA_COE_BIDDING_RESULT, "/data")
    df = unzip(zip_file_path, csv_file_pattern)

    corr = correlation(df)

    return jsonify(correlation=corr)


if __name__ == "__main__":
    app.run()
