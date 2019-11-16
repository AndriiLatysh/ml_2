import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import sklearn.metrics as sm
import sklearn.model_selection as ms


def plot_values(qualification_double_grade_df):
    plt.xlabel("English grade")
    plt.ylabel("Technical grade")

    qualified_candidates = qualification_double_grade_df[qualification_double_grade_df["qualifies"] == 1]
    unqualified_candidates = qualification_double_grade_df[qualification_double_grade_df["qualifies"] == 0]

    plt.scatter(qualified_candidates["english_grade"], qualified_candidates["technical_grade"], color="g")
    plt.scatter(unqualified_candidates["english_grade"], unqualified_candidates["technical_grade"], color="r")


def plot_model(svm_classifier):
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    plotting_step = 100
    xx = np.linspace(xlim[0], xlim[1], plotting_step)
    yy = np.linspace(ylim[0], ylim[1], plotting_step)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = svm_classifier.decision_function(xy).reshape(XX.shape)
    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(svm_classifier.support_vectors_[:, 0], svm_classifier.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none')


qualifies_double_grade = pd.read_csv("data/double_grade_reevaluated.csv")
# print(qualifies_single_grade)
plot_values(qualifies_double_grade)

plot_values(qualifies_double_grade)

X = qualifies_double_grade[["english_grade", "technical_grade"]]
y = qualifies_double_grade["qualifies"]

# svm_soft_non_linear_classifier = svm.SVC(kernel="rbf", gamma="scale")
# svm_soft_non_linear_classifier.fit(X, y)
#
# plot_model(svm_soft_non_linear_classifier)

parameter_grid = {"kernel": ["rbf"], "C": [10 ** p for p in range(-3, 5)], "gamma": [p * 1e-4 for p in range(1, 15)]}
# parameter_grid = {"kernel": ["rbf"], "C": [10 ** p for p in range(-3, 5)], "gamma": [10 ** p for p in range(-6, 2)]}

grid_search = ms.GridSearchCV(svm.SVC(), param_grid=parameter_grid, cv=4)
grid_search.fit(X, y)

print(grid_search.best_params_)

plot_model(grid_search.best_estimator_)

plt.show()
