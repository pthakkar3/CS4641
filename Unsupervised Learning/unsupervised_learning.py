import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
import sklearn.metrics as metrics
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_validate, cross_val_score

data1 = pd.read_csv('chessdata.csv', header=None, delimiter=',')
data1 = data1.as_matrix()
data2 = pd.read_csv('ttt.csv', header=None, delimiter=',')
data2 = data2.as_matrix()

chess_attributes = data1[0:, 0:35]
chess_labels = data1[0:, 36]

chess_train_x = data1[0:2499, 0:35]
chess_train_y = data1[0:2499, 36]
chess_test_x = data1[2500:, 0:35]
chess_test_y = data1[2500:, 36]

ttt_attributes = data2[0:, 0:8]
ttt_labels = data2[0:, 9]





def main():
    #Chess KMeans
    kmeans(chess_attributes, chess_labels, "Chess", 2)

    #Chess EM
    EM(chess_attributes, chess_labels, "Chess", 2)

    #TTT Kmeans
    kmeans(ttt_attributes, ttt_labels, "TTT", 2)

    #TTT EM
    EM(ttt_attributes, ttt_labels, "TTT", 2)

    #Chess PCA
    clf = PCA(random_state=0, n_components=20)
    new_att = clf.fit_transform(chess_attributes)
    new_chess_train_x = clf.fit_transform(chess_train_x)
    new_chess_test_x = clf.fit_transform(chess_test_x)

    bins = np.linspace(-.001, .001, 100)
    plt.figure()
    plt.title("Eigenvalue distribution for PCA with " + str(clf.n_components) + " components: " + "Chess")
    plt.xlabel('eigenvalue')
    plt.ylabel('frequency')
    for count, i in enumerate(clf.components_):
        plt.hist(i, bins, alpha=0.5, label=str(count + 1))

    plt.legend(loc='best')
    plt.show()

    kmeans(new_att, chess_labels, "Chess PCA", 2)
    EM(new_att, chess_labels, "Chess PCA", 2)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 2), random_state=1)
    cvscore = cross_val_score(clf, new_chess_train_x, chess_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(new_chess_train_x, chess_train_y)
    predicted = clf.predict(new_chess_test_x)
    accuracy = metrics.accuracy_score(chess_test_y, predicted)
    print("15,2 NN accuracy:", accuracy)
    plot_learning_curve(clf, "Chess PCA Neural Net 15 layers", new_chess_train_x, chess_train_y)

    # print(clf.explained_variance_ratio_)
    # print(clf.singular_values_)
    # print(clf.noise_variance_)


    #TTT PCA
    clf = PCA(random_state=0, n_components=4)
    new_att = clf.fit_transform(ttt_attributes)
    bins = np.linspace(-.001, .001, 100)
    print("TTT PCA: ", clf.components_)
    plt.figure()
    plt.title("Eigenvalue distribution for PCA with " + str(clf.n_components) + " components: " + "TTT")
    plt.xlabel('eigenvalue')
    plt.ylabel('frequency')
    for count, i in enumerate(clf.components_):
        plt.hist(i, bins, alpha=0.5, label=str(count + 1))

    plt.legend(loc='best')
    plt.show()

    kmeans(new_att, ttt_labels, "TTT PCA", 2)
    EM(new_att, ttt_labels, "TTT PCA", 2)


    #Chess ICA
    clf = FastICA(n_components=20, random_state=0)
    new_att = clf.fit_transform(chess_attributes)
    new_chess_train_x = clf.fit_transform(chess_train_x)
    new_chess_test_x = clf.fit_transform(chess_test_x)

    bins = np.linspace(-.0001, .0001, 100)
    plt.figure()
    plt.title("Components distribution for ICA with " + str(clf.n_components) + " components: " + "Chess")
    plt.xlabel('value')
    plt.ylabel('frequency')
    a = []
    for count, i in enumerate(clf.components_):
        a.extend(i)
        kurt = stats.kurtosis(i)
        plt.hist(i, bins, alpha=0.5, label=str(count + 1) + ": " + str(kurt))

    # plt.legend(loc='best')
    print(stats.kurtosis(a))
    plt.show()

    kmeans(new_att, chess_labels, "Chess ICA", 2)
    EM(new_att, chess_labels, "Chess ICA", 2)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 2), random_state=1)
    cvscore = cross_val_score(clf, new_chess_train_x, chess_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(new_chess_train_x, chess_train_y)
    predicted = clf.predict(new_chess_test_x)
    accuracy = metrics.accuracy_score(chess_test_y, predicted)
    print("15,2 NN accuracy:", accuracy)
    plot_learning_curve(clf, "Chess ICA Neural Net 15 layers", new_chess_train_x, chess_train_y)


    #TTT ICA
    clf = FastICA(n_components=4, random_state=0)
    new_att = clf.fit_transform(ttt_attributes)

    bins = np.linspace(-.0001, .0001, 100)
    plt.figure()
    plt.title("Components distribution for ICA with " + str(clf.n_components) + " components: " + "TTT")
    plt.xlabel('value')
    plt.ylabel('frequency')
    a = []
    for count, i in enumerate(clf.components_):
        a.extend(i)
        kurt = stats.kurtosis(i)
        plt.hist(i, bins, alpha=0.5, label=str(count + 1) + ": " + str(kurt))

    # plt.legend(loc='best')
    print(stats.kurtosis(a))
    plt.show()

    kmeans(new_att, ttt_labels, "TTT ICA", 2)
    EM(new_att, ttt_labels, "TTT ICA", 2)


    #RP Chess
    clf = GaussianRandomProjection(n_components=20)
    new_att = clf.fit_transform(chess_attributes)
    new_chess_train_x = clf.fit_transform(chess_train_x)
    new_chess_test_x = clf.fit_transform(chess_test_x)

    bins = np.linspace(-1, 1, 100)
    plt.figure()
    plt.title("Components distribution for RP with " + str(clf.n_components) + " components: " + "Chess")
    plt.xlabel('value')
    plt.ylabel('frequency')
    a = []
    for count, i in enumerate(clf.components_):
        a.extend(i)
        plt.hist(i, bins, alpha=0.5, label=str(count + 1))

    if clf.n_components < 10:
        plt.legend(loc='best')
    print(stats.kurtosis(a))
    plt.show()

    kmeans(new_att, chess_labels, "Chess RP", 2)
    EM(new_att, chess_labels, "Chess RP", 2)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15, 2), random_state=1)
    cvscore = cross_val_score(clf, new_chess_train_x, chess_train_y, cv=5)
    avgcv = np.mean(cvscore)
    print("CV average:", avgcv)
    clf.fit(new_chess_train_x, chess_train_y)
    predicted = clf.predict(new_chess_test_x)
    accuracy = metrics.accuracy_score(chess_test_y, predicted)
    print("15,2 NN accuracy:", accuracy)
    plot_learning_curve(clf, "Chess RP Neural Net 15 layers", new_chess_train_x, chess_train_y)


    #RP TTT
    clf = GaussianRandomProjection(n_components=4)
    new_att = clf.fit_transform(ttt_attributes)

    bins = np.linspace(-1, 1, 100)
    plt.figure()
    plt.title("Components distribution for RP with " + str(clf.n_components) + " components: " + "TTT")
    plt.xlabel('value')
    plt.ylabel('frequency')
    a = []
    for count, i in enumerate(clf.components_):
        a.extend(i)
        plt.hist(i, bins, alpha=0.5, label=str(count + 1))

    if clf.n_components < 10:
        plt.legend(loc='best')
    print(stats.kurtosis(a))
    plt.show()

    kmeans(new_att, ttt_labels, "TTT RP", 2)
    EM(new_att, ttt_labels, "TTT RP", 2)








def kmeans(attributes, labels, name, k):
    clf = KMeans(n_clusters=k, random_state=0)
    clf.fit(attributes)
    cluster_labels = clf.predict(attributes)
    encoder = LabelEncoder()
    encoder.fit(labels)
    ground_truth_labels = encoder.transform(labels)

    ar = metrics.adjusted_rand_score(ground_truth_labels, cluster_labels)
    ami = metrics.adjusted_mutual_info_score(ground_truth_labels, cluster_labels)
    hs = metrics.homogeneity_score(ground_truth_labels, cluster_labels)
    cs = metrics.completeness_score(ground_truth_labels, cluster_labels)
    ss = metrics.silhouette_score(attributes, cluster_labels)
    print(name, " KMeans ARI: ", ar)
    print(name, " KMeans AMI: ", ami)
    print(name, " KMeans Homogeneity:", hs)
    print(name, " KMeans Completeness: ", cs)
    print(name, " KMeans Silhouette: ", ss)

def EM(attributes, labels, name, k):
    clf = GaussianMixture(n_components=k, random_state=0, init_params='random')
    clf.fit(attributes)
    cluster_labels = clf.predict(attributes)
    encoder = LabelEncoder()
    encoder.fit(labels)
    ground_truth_labels = encoder.transform(labels)

    ar = metrics.adjusted_rand_score(ground_truth_labels, cluster_labels)
    ami = metrics.adjusted_mutual_info_score(ground_truth_labels, cluster_labels)
    hs = metrics.homogeneity_score(ground_truth_labels, cluster_labels)
    cs = metrics.completeness_score(ground_truth_labels, cluster_labels)
    ss = metrics.silhouette_score(attributes, cluster_labels)
    print(name, " EM ARI: ", ar)
    print(name, " EM AMI: ", ami)
    print(name, " EM Homogeneity:", hs)
    print(name, " EM Completeness: ", cs)
    print(name, " EM Silhouette: ", ss)


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