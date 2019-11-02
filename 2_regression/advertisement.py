import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.metrics as sm
import sklearn.model_selection as ms


def model_to_string(model, labels, precision=4):
    model_str = "{} = ".format(labels[0])
    for z in range(1, len(labels)):
        model_str += "{} * {} + ".format(round(model.coef_[0][z - 1], precision), labels[z])
    model_str += "{}".format(round(model.intercept_[0], precision))
    return model_str


def train_linear_model(X, y):
    linear_model = lm.LinearRegression()
    linear_model.fit(X, y)
    return linear_model


def get_MSE(model, X, y_true):
    y_predicted = model.predict(X)
    MSE = sm.mean_squared_error(y_true, y_predicted)
    return MSE


sales_df = pd.read_csv("data/advertising.csv", index_col=0)

ad_data = sales_df[["TV", "radio", "newspaper"]]
sales_data = sales_df[["sales"]]

labels = ["sales", "TV", "radio", "newspaper"]

X_train, X_test, y_train, y_test = ms.train_test_split(ad_data, sales_data, shuffle=True)

print("General model:")
sales_model = train_linear_model(X_train, y_train)
print(model_to_string(sales_model, labels))
print("Train MSE = {}".format(get_MSE(sales_model, X_train, y_train)))
print("Test MSE = {}".format(get_MSE(sales_model, X_test, y_test)))
print()

for z in range(1, 4):
    print("{} removed:".format(labels[z]))

    X_train_2_features = X_train.drop(ad_data.columns[z - 1], axis=1)
    X_test_2_features = X_test.drop(ad_data.columns[z - 1], axis=1)
    # print(X_train_2_features.head())
    labels_2_features = labels[:z] + labels[z+1:]

    model_2_features = train_linear_model(X_train_2_features, y_train)
    print(model_to_string(model_2_features, labels_2_features))
    print("MSE: {}".format(get_MSE(model_2_features, X_test_2_features, y_test)))
    print()
