### Pranshav Thakkar

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn import svm, metrics
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

# data1 = pd.read_csv('tic-tac-toe.csv', header=0, delimiter=',')
data1 = pd.read_csv('ttt.csv', header=0, delimiter=',')
data1 = data1.as_matrix()
data2 = pd.read_csv('chessdata.csv', header=0, delimiter=',')
data2 = data2.as_matrix()
# print(data2)

# for x in range(len(data2)):
#     for y in range(len(data2[x])):
#         # if isinstance(data2[x][y], float):
#         #     data2[x][y] = int(data2[x][y])
#         if data2[x][y] == 'f':
#             data2[x][y] = 0
#         elif data2[x][y] == 't':
#             data2[x][y] = 1
#         elif data2[x][y] == 'n':
#             data2[x][y] = 2
#         elif data2[x][y] == 'w':
#             data2[x][y] = 3
#         elif data2[x][y] == 'l':
#             data2[x][y] = 4
#         elif data2[x][y] == 'g':
#             data2[x][y] = 5
#         elif data2[x][y] == 'b':
#             data2[x][y] = 6

# chess = shuffle(data2)
#
# chessfile = pd.DataFrame(chess)
# chessfile.to_csv("chessdata.csv")
# for x in range(len(data1)):
#     for y in range(len(data1[x])):
#         if data1[x][y] == 'x':
#             data1[x][y] = 1
#         elif data1[x][y] == 'o':
#             data1[x][y] = 0
#         elif data1[x][y] == 'b':
#             data1[x][y] = 2
# print(data1)

# ttt = shuffle(data1)

# tttfile = pd.DataFrame(ttt)
# tttfile.to_csv("ttt.csv")


ttt_train_x = data1[0:699, 1:9]
ttt_train_y = data1[0:699, 10]
ttt_test_x = data1[700:, 1:9]
ttt_test_y = data1[700:, 10]


chess_train_x = data2[0:2499, 1:36]
chess_train_y = data2[0:2499, 37]
chess_test_x = data2[2500:, 1:36]
chess_test_y = data2[2500:, 37]



def main():

    clf = svm.SVC(kernel='linear')
    cvscore = cross_val_score(clf, ttt_train_x, ttt_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(ttt_train_x, ttt_train_y)
    predicted = clf.predict(ttt_test_x)
    accuracy = metrics.accuracy_score(ttt_test_y, predicted)
    print("Linear SVC accuracy:",accuracy)
    plot_learning_curve(clf, "TTT Linear SVM", ttt_train_x, ttt_train_y)


    clf = svm.SVC(kernel='rbf')
    cvscore = cross_val_score(clf, ttt_train_x, ttt_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(ttt_train_x, ttt_train_y)
    predicted = clf.predict(ttt_test_x)
    accuracy = metrics.accuracy_score(ttt_test_y, predicted)
    print("RBF SVC accuracy:",accuracy)
    plot_learning_curve(clf, "TTT RBF SVM", ttt_train_x, ttt_train_y)


    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
    cvscore = cross_val_score(clf, ttt_train_x, ttt_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(ttt_train_x, ttt_train_y)
    predicted = clf.predict(ttt_test_x)
    accuracy = metrics.accuracy_score(ttt_test_y, predicted)
    print("5,2 NN accuracy:",accuracy)
    plot_learning_curve(clf, "TTT Neural Net 5 layers", ttt_train_x, ttt_train_y)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,2), random_state=1)
    cvscore = cross_val_score(clf, ttt_train_x, ttt_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(ttt_train_x, ttt_train_y)
    predicted = clf.predict(ttt_test_x)
    accuracy = metrics.accuracy_score(ttt_test_y, predicted)
    print("15,2 NN accuracy:",accuracy)
    plot_learning_curve(clf, "TTT Neural Net 15 layers", ttt_train_x, ttt_train_y)

    clf = DecisionTreeClassifier(max_depth=10)
    cvscore = cross_val_score(clf, ttt_train_x, ttt_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(ttt_train_x, ttt_train_y)
    predicted = clf.predict(ttt_test_x)
    accuracy = metrics.accuracy_score(ttt_test_y, predicted)
    print("Decision Tree accuracy:",accuracy)
    plot_learning_curve(clf, "TTT Decision Tree", ttt_train_x, ttt_train_y)


    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=50)
    cvscore = cross_val_score(clf, ttt_train_x, ttt_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(ttt_train_x, ttt_train_y)
    predicted = clf.predict(ttt_test_x)
    accuracy = metrics.accuracy_score(ttt_test_y, predicted)
    print("Boosted Tree accuracy:",accuracy)
    plot_learning_curve(clf, "TTT Boosted Tree", ttt_train_x, ttt_train_y)


    clf = KNeighborsClassifier(n_neighbors=3)
    cvscore = cross_val_score(clf, ttt_train_x, ttt_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(ttt_train_x, ttt_train_y)
    predicted = clf.predict(ttt_test_x)
    accuracy = metrics.accuracy_score(ttt_test_y, predicted)
    print("3 KNN accuracy:",accuracy)
    plot_learning_curve(clf, "TTT 3 Nearest Neighbor", ttt_train_x, ttt_train_y)


    clf = KNeighborsClassifier(n_neighbors=5)
    cvscore = cross_val_score(clf, ttt_train_x, ttt_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(ttt_train_x, ttt_train_y)
    predicted = clf.predict(ttt_test_x)
    accuracy = metrics.accuracy_score(ttt_test_y, predicted)
    print("5 KNN accuracy:",accuracy)
    plot_learning_curve(clf, "TTT 5 Nearest Neighbor", ttt_train_x, ttt_train_y)


    clf = KNeighborsClassifier(n_neighbors=7)
    cvscore = cross_val_score(clf, ttt_train_x, ttt_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(ttt_train_x, ttt_train_y)
    predicted = clf.predict(ttt_test_x)
    accuracy = metrics.accuracy_score(ttt_test_y, predicted)
    print("7 KNN accuracy:",accuracy)
    plot_learning_curve(clf, "TTT 7 Nearest Neighbor", ttt_train_x, ttt_train_y)


    clf = svm.SVC(kernel='linear')
    cvscore = cross_val_score(clf, chess_train_x, chess_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(chess_train_x, chess_train_y)
    predicted = clf.predict(chess_test_x)
    accuracy = metrics.accuracy_score(chess_test_y, predicted)
    print("Chess Linear SVC accuracy:",accuracy)
    plot_learning_curve(clf, "Chess Linear SVM", chess_train_x, chess_train_y)


    clf = svm.SVC(kernel='rbf')
    cvscore = cross_val_score(clf, chess_train_x, chess_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(chess_train_x, chess_train_y)
    predicted = clf.predict(chess_test_x)
    accuracy = metrics.accuracy_score(chess_test_y, predicted)
    print("Chess RBF SVC accuracy:",accuracy)
    plot_learning_curve(clf, "Chess RBF SVM", chess_train_x, chess_train_y)


    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1)
    cvscore = cross_val_score(clf, chess_train_x, chess_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(chess_train_x, chess_train_y)
    predicted = clf.predict(chess_test_x)
    accuracy = metrics.accuracy_score(chess_test_y, predicted)
    print("Chess 5,2 NN accuracy:",accuracy)
    plot_learning_curve(clf, "Chess 5 Layer Neural Net ", chess_train_x, chess_train_y)


    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,2), random_state=1)
    cvscore = cross_val_score(clf, chess_train_x, chess_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(chess_train_x, chess_train_y)
    predicted = clf.predict(chess_test_x)
    accuracy = metrics.accuracy_score(chess_test_y, predicted)
    print("Chess 15,2 NN accuracy:",accuracy)
    plot_learning_curve(clf, "Chess 15 Layer Neural Net", chess_train_x, chess_train_y)


    clf = DecisionTreeClassifier(max_depth=10)
    cvscore = cross_val_score(clf, chess_train_x, chess_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(chess_train_x, chess_train_y)
    predicted = clf.predict(chess_test_x)
    accuracy = metrics.accuracy_score(chess_test_y, predicted)
    print("Chess Decision Tree accuracy:",accuracy)
    plot_learning_curve(clf, "Chess Decision Tree", chess_train_x, chess_train_y)


    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=5), n_estimators=50)
    cvscore = cross_val_score(clf, chess_train_x, chess_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(chess_train_x, chess_train_y)
    predicted = clf.predict(chess_test_x)
    accuracy = metrics.accuracy_score(chess_test_y, predicted)
    print("Chess Boosted Tree accuracy:",accuracy)
    plot_learning_curve(clf, "Chess Boosted Tree", chess_train_x, chess_train_y)


    clf = KNeighborsClassifier(n_neighbors=3)
    cvscore = cross_val_score(clf, chess_train_x, chess_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(chess_train_x, chess_train_y)
    predicted = clf.predict(chess_test_x)
    accuracy = metrics.accuracy_score(chess_test_y, predicted)
    print("Chess 3 KNN accuracy:",accuracy)
    plot_learning_curve(clf, "Chess 3 Nearest Neighbors", chess_train_x, chess_train_y)


    clf = KNeighborsClassifier(n_neighbors=5)
    cvscore = cross_val_score(clf, chess_train_x, chess_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(chess_train_x, chess_train_y)
    predicted = clf.predict(chess_test_x)
    accuracy = metrics.accuracy_score(chess_test_y, predicted)
    print("Chess 5 KNN accuracy:",accuracy)
    plot_learning_curve(clf, "Chess 5 Nearest Neighbors", chess_train_x, chess_train_y)


    clf = KNeighborsClassifier(n_neighbors=7)
    cvscore = cross_val_score(clf, chess_train_x, chess_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(chess_train_x, chess_train_y)
    predicted = clf.predict(chess_test_x)
    accuracy = metrics.accuracy_score(chess_test_y, predicted)
    print("Chess 7 KNN accuracy:",accuracy)
    plot_learning_curve(clf, "Chess 7 Nearest Neighbors", chess_train_x, chess_train_y)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt

if __name__ == '__main__':
    main()