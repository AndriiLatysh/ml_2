import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm


def model_to_string(model, labels, precision=4):
    model_str = "{} = ".format(labels[0])
    for z in range(1, len(labels)):
        model_str += "{} * {} + ".format(round(model.coef_.flatten()[z - 1], precision), labels[z])
    model_str += "{}".format(round(model.intercept_[0], precision))
    return model_str


sales_df = pd.read_csv("data/advertising.csv", index_col=0)

X = sales_df[["TV", "radio", "newspaper"]]
y = sales_df[["sales"]]

labels = ["sales", "TV", "radio", "newspaper"]

linear_regression = lm.LinearRegression()
lasso_regression = lm.Lasso()
ridge_regression = lm.Ridge()

linear_regression.fit(X, y)
lasso_regression.fit(X, y)
ridge_regression.fit(X, y)

print("Linear: ")
print(model_to_string(linear_regression, labels))
print()

print("L1: ")
print(model_to_string(lasso_regression, labels))
print()

print("L2: ")
print(model_to_string(ridge_regression, labels))
print()
