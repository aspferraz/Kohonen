import numpy as np
import pandas as pd
import copy
from sklearn import metrics


def getNeighborhood(A, wCoords, r=1):
    x = wCoords[0]
    y = wCoords[1]

    column = A[:, y]
    row = A[x, :]

    xToEnd = (len(row) - 1) - y
    xToStart = y
    yToEnd = (len(column) - 1) - x
    yToStart = x

    n = ((x - r, x), y) if r <= yToStart else ((x - yToStart, x), y)
    s = ((x + 1, x + (r + 1)), y) if r <= yToEnd else ((x + 1, x + (yToEnd + 1)), y)

    e = (x, (y + 1, y + (r + 1))) if r <= xToEnd else (x, (y + 1, y + (xToEnd + 1)))
    w = (x, (y - r, y)) if r <= xToStart else (x, (y - xToStart, y))

    return n, e, s, w


def updateNeighborhood(A, sample, nCoords, alpha):
    x = nCoords[1][0]
    y = nCoords[0][1]

    column = A[:, y]
    row = A[x, :]

    nC = nCoords[0][0]
    sC = nCoords[2][0]
    eC = nCoords[1][1]
    wC = nCoords[3][1]

    n = column[nC[0]: nC[1]]
    s = column[sC[0]: sC[1]]
    e = row[eC[0]: eC[1]]
    w = row[wC[0]: wC[1]]

    n = n + (alpha / 2) * (sample - n)
    column[nC[0]: nC[1]] = n

    s = s + (alpha / 2) * (sample - s)
    column[sC[0]: sC[1]] = s
    e = e + (alpha / 2) * (sample - e)
    row[eC[0]: eC[1]] = e
    w = w + (alpha / 2) * (sample - w)
    row[wC[0]: wC[1]] = w

    A[:, y] = column
    A[x, :] = row


nodes = 4


# for u_node in itertools.product(range(nodes * 2 - 1),
#                                         range(nodes * 2 - 1)):
#
#     print(u_node[0], u_node[1])
#
# u_matrix = np.zeros(
#             shape=(nodes * 2 - 1, nodes * 2 - 1, 1),
#             dtype=float)
# print(u_matrix)

def findBestRelation(cluster, numClusters, pairs):
    # TODO: this dict should be created dynamically
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

classes = [0, 1, 0, 0, 1, 2, 0, 2, 1, 0, 1, 1, 2, 2, 2]
clusters = [1, 2, 1, 1, 2, 0, 1, 0, 2, 1, 2, 1, 0, 0, 2]

results = np.asarray([classes, clusters]).T
results = results[np.argsort(results[:, 0])]

print(results)

column_values = ['classes', 'clusters']
data = pd.DataFrame(data=results, columns=column_values)

# print(data)

# print(findBestRelation(1, 3, results))
print(calcAccuracy(3, results))
