import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.metrics as sm
import sklearn.model_selection as ms


def plot_model(model):
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
            prediction_point = [[x, y]]
            probability_level[x, y] = model.predict_proba(prediction_point)[:, 1]

    plt.contourf(probability_level, cmap="rainbow")

    plt.scatter(qualified_candidates["technical_grade"], qualified_candidates["english_grade"], color="w")
    plt.scatter(unqualified_candidates["technical_grade"], unqualified_candidates["english_grade"], color="k")


qualifies_double_grade = pd.read_csv("data/double_grade.csv")
# print(qualifies_single_grade)

X = qualifies_double_grade[["technical_grade", "english_grade"]]
y = qualifies_double_grade["qualifies"]

number_of_folds = 4

cv_qualification_model = lm.LogisticRegression(solver="lbfgs")
# cv_qualification_model = lm.LogisticRegression()

cv_model_quality = ms.cross_val_score(cv_qualification_model, X, y, cv=number_of_folds, scoring="accuracy")
print(cv_model_quality)

cv_model_prediction = ms.cross_val_predict(cv_qualification_model, X, y, cv=number_of_folds)
cv_confusion_matrix = sm.confusion_matrix(y, cv_model_prediction)
print(cv_confusion_matrix)

qualification_model = lm.LogisticRegression(solver="lbfgs")
qualification_model.fit(X, y)

modeled_qualification_probability = qualification_model.predict_proba(X)[:, 1]
qualifies_double_grade["probability"] = modeled_qualification_probability

qualifies_double_grade.sort_values(by="probability", inplace=True)
pd.set_option("display.max_rows", None)

print(qualifies_double_grade)

print(qualification_model.coef_)
print(qualification_model.intercept_)

plot_model(qualification_model)

plt.show()
