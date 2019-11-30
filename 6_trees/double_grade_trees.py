import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.tree as sk_trees
import sklearn.model_selection as sk_ms
import sklearn.metrics as sk_metrics
import sklearn.ensemble as sk_ensemble


def plot_model(model, qualifies_double_grade):
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


qualification_double_grade_df = pd.read_csv("data/double_grade_reevaluated.csv")

X = qualification_double_grade_df[["english_grade", "technical_grade"]]
y = qualification_double_grade_df["qualifies"]

forest_classifier = sk_ensemble.RandomForestClassifier()
forest_classifier.fit(X, y)

plot_model(forest_classifier, qualification_double_grade_df)

plt.show()
