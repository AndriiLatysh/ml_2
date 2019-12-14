import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neural_network as sk_nn
import sklearn.preprocessing as sk_preprocessing
import sklearn.metrics as sk_metrics
import sklearn.model_selection as sk_ms


def plot_model(model, min_max_scaler):
    plt.xlabel("Technical grade")
    plt.ylabel("English grade")

    qualified_candidates = qualifies_double_grade[qualifies_double_grade["qualifies"] == 1]
    unqualified_candidates = qualifies_double_grade[qualifies_double_grade["qualifies"] == 0]

    max_grade = 101
    english_grades_range = list(range(max_grade))
    technical_grades_range = list(range(max_grade))
    probability_level = np.empty([max_grade, max_grade])

    for x in technical_grades_range:
        for y in english_grades_range:
            prediction_point = min_max_scaler.transform([[x, y]])
            probability_level[x, y] = model.predict_proba(prediction_point)[:, 1]

    plt.contourf(probability_level, cmap="rainbow")

    plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="w")
    plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="k")


qualifies_double_grade = pd.read_csv("data/double_grade_reevaluated.csv")

X = qualifies_double_grade[["technical_grade", "english_grade"]]
y = qualifies_double_grade[["qualifies"]]

min_max_scaler = sk_preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

one_hot_encoding = sk_preprocessing.OneHotEncoder(sparse=False)
y = one_hot_encoding.fit_transform(y)

X_train, X_test, y_train, y_test = sk_ms.train_test_split(X, y)

qualification_model = sk_nn.MLPClassifier(hidden_layer_sizes=(6, 6), max_iter=1000000)
qualification_model.fit(X_train, y_train)

y_predict = qualification_model.predict(X_test)

transformed_y_test = one_hot_encoding.inverse_transform(y_test)
transformed_y_predict = one_hot_encoding.inverse_transform(y_predict)

confusion_matrix = sk_metrics.confusion_matrix(transformed_y_test, transformed_y_predict)

print(confusion_matrix)

plot_model(qualification_model, min_max_scaler)
plt.show()
