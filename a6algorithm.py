"""
Primary algorithm for k-Means clustering

This file contains the Algorithm class for performing k-means clustering.  While it is
the last part of the assignment, it is the heart of the clustering algorithm.  You
need this class to view the complete visualizer.

Paliska Bradley, bp355 and Philip Coppolino, pcc78
11/18/2021
"""
import math
import random
import numpy


# For accessing the previous parts of the assignment
import a6dataset
import a6cluster


# Part A
def valid_seeds(value, size):
    """
    Returns True if value is a valid list of seeds for clustering.

    A list of seeds is a k-element list OR tuple of integersa between 0 and size-1.
    In addition, no seed element can appear twice.

    Parameter value: a value to check
    Precondition: value can be anything

    Paramater size: The database size
    Precondition: size is an int > 0
    """
    if (type(size) != int) and size <= 0:
        return False
    if (type(value) != list):
        return False
    for j in value:
        if (type(j) != int):
            return False
        if j >= size:
            return False
    x = []
    for i in value:
        if i not in x:
            x.append(i)
    if x != value:
        return False

    return True


class Algorithm(object):
    """
    A class to manage and run the k-means algorithm.

    The method step() performs one step of the calculation.  The method run() will
    continue the calculation until it converges (or reaches a maximum number of steps).
    """
    # IMMUTABLE ATTRIBUTES (Fixed after initialization with no DIRECT access)
    # Attribute _dataset: The Dataset for this algorithm
    # Invariant: _dataset is an instance of Dataset
    #
    # Attribute _cluster: The clusters to use at each step
    # Invariant: _cluster is a non-empty list of Cluster instances


    # Part B
    def getClusters(self):
        """
        Returns the list of clusters in this object.

        This method returns the cluster list directly (it does not copy).  Any changes
        made to this list will modify the set of clusters.
        """
        return self._cluster


    def __init__(self, dset, k, seeds=None):
        """
        Initializes the algorithm for the dataset ds, using k clusters.

        If the optional argument seeds is supplied, those seeds will be a list OR
        tuple of indices into the dataset. They specify which points should be the
        initial cluster centroids. Otherwise, the clusters are initialized by randomly
        selecting k different points from the database to be the cluster centroids.

        Parameter dset: the dataset
        Precondition: dset is an instance of Dataset

        Parameter k: the number of clusters
        Precondition: k is an int, 0 < k <= dset.getSize()

        Paramter seeds: the initial cluster indices (OPTIONAL)
        Precondition: seeds is None, or a list/tuple of valid seeds.
        """
        assert isinstance(dset, a6dataset.Dataset)
        assert type(k) == int
        assert k > 0 and k <= dset.getSize()
        assert seeds == None or valid_seeds(seeds, dset.getSize())
        self._dataset = dset


        if seeds == None:
            seeds = random.sample(range(self._dataset.getSize()), k)

        self._cluster = []
        for i in seeds:
            self._cluster += [a6cluster.Cluster(dset, self._dataset.getPoint(i))]


    # Part C
    def _nearest(self, point):
        """
        Returns the cluster nearest to point

        This method uses the distance method of each Cluster to compute the distance
        between point and the cluster centroid. It returns the Cluster that is closest.

        Ties are broken in favor of clusters occurring earlier in the list returned
        by getClusters().

        Parameter point: The point to compare.
        Precondition: point is a tuple of numbers (int or float). Its length is the
        same as the dataset dimension.
        """
        assert a6dataset.is_point(point)
        assert len(point) == self._dataset.getDimension()
        x = self._cluster[0]
        fdis = self._cluster[0].distance(point)
        for i in self._cluster:
            dis = i.distance(point)
            if fdis > dis:
                x = i
                fdis = dis
        return x


    def _partition(self):
        """
        Repartitions the dataset so each point is in exactly one Cluster.
        """
        for j in self._cluster:
            j.clear()
        for i in range(len(self._dataset.getContents())):
            x = self._nearest(self._dataset.getPoint(i))
            x.addIndex(i)


    # Part D
    def _update(self):
        """
        Returns True if all centroids are unchanged after an update; False otherwise.

        This method first updates the centroids of all clusters'.  When it is done, it
        checks whether any of them have changed. It returns False if just one has
        changed. Otherwise, it returns True.
        """
        result = True
        for i in self._cluster:
            if i.update() == False:
                result = False
        return result


    def step(self):
        """
        Returns True if the algorithm converges after one step; False otherwise.

        This method performs one cycle of the k-means algorithm. It then checks if
        the algorithm has converged and returns the appropriate result (True if
        converged, false otherwise).
        """
        self._partition()
        return self._update()


    # Part D
    def run(self, maxstep):
        """
        Continues clustering until either it converges or performs maxstep steps.

        After the maxstep call to step, if this calculation did not converge, this
        method will stop.

        Parameter maxstep: The maximum number of steps to perform
        Precondition: maxstep is an int >= 0
        """
        assert type(maxstep) == int
        for i in range(maxstep):
            if self.step() == True:
                self.step()
                return
            self.step()
