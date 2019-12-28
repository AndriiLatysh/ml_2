import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as sk_trees
import sklearn.model_selection as sk_ms
import sklearn.metrics as sk_metrics
import sklearn.ensemble as sk_ensemble


plt.figure(figsize=(18, 10))
diabetes_df = pd.read_csv("data/pima-indians-diabetes.csv")
column_names = diabetes_df.columns.tolist()
# print(column_names)
# print(len(diabetes_df))

X = diabetes_df[column_names[:-1]]
y = diabetes_df[column_names[-1]]

# print(X.head())
# print(y.head())

X_train, X_test, y_train, y_test = sk_ms.train_test_split(X, y)

diabetes_adaboost_model = sk_ensemble.AdaBoostClassifier(n_estimators=100)

diabetes_adaboost_model.fit(X_train, y_train)

y_prediction = diabetes_adaboost_model.predict(X_test)

print("Accuracy: {}".format(sk_metrics.accuracy_score(y_test, y_prediction)))

confusion_matrix = sk_metrics.confusion_matrix(y_test, y_prediction)
print(confusion_matrix)

