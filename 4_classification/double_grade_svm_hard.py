import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm


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


qualification_double_grade_df = pd.read_csv("data/double_grade_small.csv")

plot_values(qualification_double_grade_df)

X = qualification_double_grade_df[["english_grade", "technical_grade"]]
y = qualification_double_grade_df["qualifies"]

svm_hard_linear_classifier = svm.SVC(kernel="linear")
svm_hard_linear_classifier.fit(X, y)

plot_model(svm_hard_linear_classifier)

plt.show()
