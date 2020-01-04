import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn.metrics as sk_metrics
import statsmodels.tsa.seasonal as statsmodels_seasonal
import statsmodels.tsa.statespace.sarimax as statmodels_sarimax
import fbprophet as prophet


def create_data_frame(values, last_date):
    dates = pd.date_range(start=last_date+pd.DateOffset(months=1), periods=len(values), freq="MS")
    predicted_df = pd.DataFrame({"date": dates, "value": values})
    return predicted_df


def naive_prediction(train_df, observation_to_predict):
    values = [train_df.iat[-1, 1] for i in range(observation_to_predict)]
    return create_data_frame(values, train_df.iat[-1, 0])


def average_prediction(train_df, observation_to_predict):
    values = [train_df["value"].mean() for i in range(observation_to_predict)]
    return create_data_frame(values, train_df.iat[-1, 0])


def sarima_prediction(train_df, observation_to_predict):
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 12)
    sarima_model = statmodels_sarimax.SARIMAX(train_df["value"], order=order, seasonal_order=seasonal_order)
    sarima_model_fit = sarima_model.fit(disp=False)
    values = sarima_model_fit.forecast(observation_to_predict)
    return create_data_frame(values, train_df.iat[-1, 0])


def prophet_prediction(train_df, observation_to_predict):
    prophet_model = prophet.Prophet(yearly_seasonality=True)
    prophet_model.fit(train_df.rename(columns={"date": "ds", "value": "y"}))
    future = prophet_model.make_future_dataframe(periods=observation_to_predict)
    forecast = prophet_model.predict(future)
    values = forecast["yhat"].iloc[-observation_to_predict:]
    return create_data_frame(values, train_df.iat[-1, 0])


def make_cv_splits(data_df, splits):
    cv_splits = []
    split_size = len(data_df) // splits
    for z in range(2*split_size, len(data_df), split_size):
        train_df = data_df.iloc[0:z]
        test_df = data_df.iloc[z:z+split_size]
        cv_splits.append((train_df, test_df))
    return cv_splits


def make_cv_predictions(cv_splits, model):
    predictions = []
    for train_df, test_df in cv_splits:
        predicted_df = model(train_df, len(test_df))
        predictions.append(predicted_df)
    return predictions


def get_cv_errors(cv_splits, predictions):
    errors = {"MAE": [], "RMSLE": []}
    for z in range(len(predictions)):
        test_df = cv_splits[z][1]
        predicted_df = predictions[z]
        errors["MAE"].append(sk_metrics.mean_absolute_error(test_df["value"], predicted_df["value"]))
        errors["RMSLE"].append(math.sqrt(sk_metrics.mean_squared_log_error(test_df["value"], predicted_df["value"])))
    for error_type, error_list in errors.items():
        errors[error_type] = np.mean(error_list)
    return errors


def plot_cv_prediction(predictions):
    for prediction in predictions:
        plt.plot(prediction["date"], prediction["value"], color="y")

# def cross_validate(data_df, splits, model):
#     split_size = len(data_df) // splits
#     errors = {"MAE": [], "RMSLE": []}
#     for z in range(2*split_size, len(data_df), split_size):
#         train_df = data_df.iloc[0:z]
#         test_df = data_df.iloc[z:z+split_size]
#         predicted_df = model(train_df, split_size)
#         errors["MAE"].append(sk_metrics.mean_absolute_error(test_df["value"], predicted_df["value"]))
#         errors["RMSLE"].append(math.sqrt(sk_metrics.mean_squared_log_error(test_df["value"], predicted_df["value"])))
#         # plt.plot(predicted_df["date"], predicted_df["value"], color="y")
#     for error_type, error_list in errors.items():
#         errors[error_type] = np.mean(error_list)
#     return errors

plt.figure(figsize=(20, 10))

utility_index_df = pd.read_csv("data/IPG2211A2N.csv", parse_dates=["DATE"])
utility_index_df.rename(columns={"DATE": "date", "IPG2211A2N": "value"}, inplace=True)

utility_index_df = utility_index_df[
    (utility_index_df["date"] >= pd.Timestamp("1979-01-01")) & (utility_index_df["date"] < pd.Timestamp("2019-01-01"))]

print(len(utility_index_df))
print(utility_index_df["date"].min())
print(utility_index_df["date"].max())

plt.plot(utility_index_df["date"], utility_index_df["value"])

cross_validation_splits = 6
cv_splits = make_cv_splits(utility_index_df, cross_validation_splits)

naive_predictions = make_cv_predictions(cv_splits, naive_prediction)
naive_errors = get_cv_errors(cv_splits, naive_predictions)
print("Naive errors:", naive_errors)

average_predictions = make_cv_predictions(cv_splits, average_prediction)
average_errors = get_cv_errors(cv_splits, average_predictions)
print("Average errors:", average_errors)

sarima_predictions = make_cv_predictions(cv_splits, sarima_prediction)
sarima_errors = get_cv_errors(cv_splits, sarima_predictions)
print("SARIMA errors:", sarima_errors)
plot_cv_prediction(sarima_predictions)

sarima_extrapolation = sarima_prediction(utility_index_df, 80)
plt.plot(sarima_extrapolation["date"], sarima_extrapolation["value"], color="g")

prophet_predictions = make_cv_predictions(cv_splits, prophet_prediction)
prophet_errors = get_cv_errors(cv_splits, prophet_predictions)
print("Prophet errors:", prophet_errors)
plot_cv_prediction(prophet_predictions)

prophet_extrapolation = prophet_prediction(utility_index_df, 80)
plt.plot(prophet_extrapolation["date"], prophet_extrapolation["value"], color="r")

# SARIMA errors: {'MAE': 3.1180758431140116, 'RMSLE': 0.039720171116621746}

# print("Naive", cross_validate(utility_index_df, cross_validation_splits, naive_prediction))
# print("Average", cross_validate(utility_index_df, cross_validation_splits, average_prediction))

utility_index_decomposition = statsmodels_seasonal.seasonal_decompose(utility_index_df["value"], model="additive", freq=12)
utility_index_decomposition.plot()

plt.show()
