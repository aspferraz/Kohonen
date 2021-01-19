import warnings
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kohonen import Kohonen
# import susi
warnings.simplefilter(action='ignore', category=FutureWarning)


def covertIrisClass(className):
    if className == str('iris-setosa'):
        return 0
    elif className == str('iris-versicolor'):
        return 1
    elif className == str('iris-virginica'):
        return 2


def getDataFrame(filePath, skipNum=9, sepChars=',\s'):
    df1 = pd.read_table(filePath, header=None, skiprows=skipNum, sep=sepChars, engine='python')
    class_codes = [covertIrisClass(s) for s in df1[4].str.lower()]
    df2 = pd.DataFrame(class_codes)
    df3 = pd.concat([df1, df2], axis=1, ignore_index=True)
    df3 = df3.iloc[np.random.permutation(len(df3))]
    return df3


def findBestRelation(cluster, numClusters, pairs):
    # TODO: this dict must be created dynamically
    votes = {
        # (c, k): v,
        (0, 0): 0,
        (1, 0): 0,
        (2, 0): 0,
        (0, 1): 0,
        (1, 1): 0,
        (2, 1): 0,
        (0, 2): 0,
        (1, 2): 0,
        (2, 2): 0,
    }

    for i in range(len(pairs)):
        c = pairs[i, 0]
        k = pairs[i, 1]
        votes[(c, k)] += 1

    # sorting by value, desc
    votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)

    # get the best relations
    bestRelations = {key: votes[key] for key in range(0, numClusters)}
    sel = None
    for b in bestRelations.values():
        if b[0][1] == cluster:
            if sel is not None:
                if sel[1] < b[1]:
                    sel = b
            else:
                sel = b

    clazz = None
    if sel is not None:
        clazz = sel[0][0]

    return clazz


def calcAccuracy(numClusters, pairs):
    d = copy.deepcopy(pairs)
    for p in d:
        c = findBestRelation(p[1], numClusters, pairs)
        if c is not None:
            p[1] = c

    accuracy = metrics.accuracy_score(list(d[:, 0]), list(d[:, 1]))
    return accuracy


def run():
    # Training #

    # clf = susi.SOMClustering(n_rows=50, n_columns=50, learning_rate_start=0.5, learning_rate_end=0.01)
    # clf.fit(X)
    # u = clf.get_u_matrix()
    # w = clf.unsuper_som_

    clf, w_, w = Kohonen(gridWidth, maxEpochs, alphaStart, alphaEnd).fit(X)
    u = clf.getUMatrix(w)

    plt.imshow(np.squeeze(u), cmap="Greys")
    plt.colorbar()
    plt.show()

    # Tests #

    # results = np.array(clf.get_bmus(X_t, w))
    results = clf.classify(X_t, w)

    plt.scatter(results[:, 0], results[:, 1])
    plt.xlim(0, gridWidth)  # x range
    plt.ylim(0, gridWidth)  # y range
    plt.grid()
    plt.show()

    # K-Means
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300,
                    algorithm='full')  # The classical EM-style algorithm is 'full'

    clusters = kmeans.fit_predict(results)
    plt.scatter(results[:, 0], results[:, 1], c=clusters)
    plt.xlim(0, gridWidth)
    plt.ylim(0, gridWidth)
    plt.grid()
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=70, c='red')
    plt.show()

    # Accuracy calc
    results = np.asarray([Y_t_, Y_t, clusters]).T
    results = results[np.argsort(results[:, 1])]

    column_values = ['class names', 'classes', 'clusters']
    df = pd.DataFrame(data=results,
                      columns=column_values)
    print(df)

    print(f'\nAccuracy: {calcAccuracy(3, results[:, 1:3])}')


if __name__ == '__main__':
    df_tra = getDataFrame('iris/10-fold/iris-10-10tra.dat')
    df_tst = getDataFrame('iris/10-fold/iris-10-10tst.dat')

    # Define the scaler for normalization
    scaler = StandardScaler().fit(df_tra.iloc[:, 0:4])

    X = scaler.transform(df_tra.iloc[:, 0:4])
    X_t = scaler.transform(df_tst.iloc[:, 0:4])
    Y_t = np.asarray(df_tst.iloc[:, 5])
    Y_t_ = np.asarray(df_tst.iloc[:, 4])

    maxEpochs = 1000
    alphaStart = 0.05
    alphaEnd = 0.001

    gridWidth = 3  # grids (4x4, 15x15, 40x40)
    run()
