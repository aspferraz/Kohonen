# Kohonen

Implementation of a Kohonen Network in Python, applied to the Iris Plants dataset, available in Keel Dataset
(https://sci2s.ugr.es/keel/datasets.php).

The network is trained considering three different topologies, with a rate of
learning = 0.001, with the topological grid being two-dimensional, having a neighborhood radius
between neurons equal to 1.

For each topology, the U-matrix is generated and the K-means algorithm is performed, using K = 3 and measure of
Euclidean distance.

The chosen network is tested, from the cluster centers found by the K-means algorithm,
with the test set. Finally, the network is evaluated for the formation of the groups, that is, it is verified whether the data of
tests were organized into three classes: Iris-setosa, Iris-versicolor, Iris-virginica.
