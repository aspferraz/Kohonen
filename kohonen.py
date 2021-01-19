'''
    File name: kohonen.py
    Author: Antonio Ferraz
    Date created: 10/18/2020
    Date last modified: 10/23/2020
    Python Version: 3.8
'''
import numpy as np
from numpy.random import rand
import copy
import itertools

class Kohonen(object):

    def __init__(self, gridWidth, epochs, alphaStart=0.5, alphaEnd=0.001):
        self.gridWidth = gridWidth
        self.epochs = epochs
        self.alphaStart = alphaStart
        self.alphaEnd = alphaEnd
        # init radius parameters
        self.radiusMax = max(gridWidth, gridWidth) / 2
        self.radiusMin = 1

    # Return the position of the winner node
    def getWinner(self, A, s):
        a = np.linalg.norm(A - s.astype(np.float64), axis=2)
        return np.argwhere(a == np.min(a))[0]

    def fit(self, X, w=None):
        if w is None:
            w = []

        self.nodeList = np.array(list(
            itertools.product(range(self.gridWidth), range(self.gridWidth))),
            dtype=int)

        self.weights = np.random.rand(self.gridWidth, self.gridWidth, X.shape[1]) \
            if len(w) == 0 \
            else w

        initialWeights = copy.deepcopy(self.weights)  # makes an copy of the original weights

        sampleForCalc = np.full(
            fill_value=1., shape=(len(X), 1))

        winners = []

        for epoch in range(1, self.epochs):
            lastWinners = winners
            winners = []

            for j in range(X.shape[0]):
                # training sample
                sample = X[j]

                # Compute winner vector
                win = self.getWinner(self.weights, sample)
                winners.append(win)

                # Update winner and its neighbors
                r = self.getCurrentRadius(epoch)
                a = self.getCurrentAlpha(epoch)
                # self.update(weights, sample, win, r, a)

                # calculate distance weight matrix and update weights
                nbhWeights = self.getNbhDistanceWeights(r, win)

                self.weights = self.modifyWeights(
                    self.weights, nbhWeights,
                    sample=sample,
                    alpha=a * sampleForCalc[j])

            if epoch % 25 == 0 or epoch == 1:
                print('Epoch ', epoch)

            if np.array_equiv(np.array(winners), np.array(lastWinners)):
                print('SOM trained in ' + str(epoch) + ' epochs')
                break

        return self, initialWeights, self.weights

    def modifyWeights(self, weights, nbhDistanceWeights, sample, alpha):
        return weights + np.multiply(alpha, np.multiply(
            nbhDistanceWeights, -np.subtract(weights, sample)))

    def getNbhDistanceWeights(self, radius, winnerPos):
        distance = np.linalg.norm(self.nodeList - winnerPos, axis=1)
        pseudoGaussian = np.exp(-np.divide(np.power(distance, 2),
                                           (2 * np.power(radius, 2))))

        nbhDistWeights = pseudoGaussian.reshape((self.gridWidth, self.gridWidth, 1))

        return nbhDistWeights

    def getCurrentAlpha(self, iteration):
        # min
        return self.alphaStart * np.power(self.alphaEnd / self.alphaStart, iteration / self.epochs)

    def getCurrentRadius(self, iteration):
        # linear
        return (self.radiusMin - self.radiusMax) * (1 - np.divide(iteration, self.epochs)) + self.radiusMax

    def classify(self, X, weights):
        results = []

        for j in range(X.shape[0]):
            # test sample
            sample = X[j]
            # classify test sample
            win = self.getWinner(weights, sample)
            results.append(np.asarray(win))

        return np.asarray(results)

    def getUMatrix(self, weights):
        self.weights = weights
        self.uMatrix = np.zeros(
            shape=(self.gridWidth * 2 - 1, self.gridWidth * 2 - 1, 1),
            dtype=float)

        # set values between SOM nodes
        self.calcUMatrixDistances()

        # set values at SOM nodes and on diagonals
        self.calcUMatrixMeans()

        return self.uMatrix

    def calcUMatrixDistances(self):
        for u in itertools.product(range(self.gridWidth * 2 - 1),
                                   range(self.gridWidth * 2 - 1)):

            if not (u[0] % 2) and (u[1] % 2):
                # mean horizontally
                self.uMatrix[u] = np.linalg.norm(
                    self.weights[u[0] // 2][u[1] // 2] -
                    self.weights[u[0] // 2][u[1] // 2 + 1])
            elif (u[0] % 2) and not (u[1] % 2):
                # mean vertically
                self.uMatrix[u] = np.linalg.norm(
                    self.weights[u[0] // 2][u[1] // 2] -
                    self.weights[u[0] // 2 + 1][u[1] // 2],
                    axis=0)

    def calcUMatrixMeans(self):
        for u in itertools.product(range(self.gridWidth * 2 - 1),
                                   range(self.gridWidth * 2 - 1)):

            nodes = []
            if not (u[0] % 2) and not (u[1] % 2):
                # SOM nodes -> mean over 2-4 values

                if u[0] > 0:
                    nodes.append((u[0] - 1, u[1]))
                if u[0] < self.gridWidth * 2 - 2:
                    nodes.append((u[0] + 1, u[1]))
                if u[1] > 0:
                    nodes.append((u[0], u[1] - 1))
                if u[1] < self.gridWidth * 2 - 2:
                    nodes.append((u[0], u[1] + 1))

            elif (u[0] % 2) and (u[1] % 2):
                # mean over four
                nodes = ([
                    (u[0] - 1, u[1]),
                    (u[0] + 1, u[1]),
                    (u[0], u[1] - 1),
                    (u[0], u[1] + 1)])

            if nodes:
                self.uMatrix[u] = np.mean([self.uMatrix[u_node] for u_node in nodes])
