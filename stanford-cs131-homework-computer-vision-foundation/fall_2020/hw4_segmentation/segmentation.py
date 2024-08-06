"""
CS131 - Computer Vision: Foundations and Applications
Assignment 4
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 10/9/2020
Python Version: 3.5+
"""

import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(data, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        data - Each row represents a feature vector. N rows of data. N*2
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster j = assignments[i])
    """

    N, D = data.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # 1. Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = data[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for _ in range(num_iters):
        
        # 2. Assign each point to the closest center
        distance = cdist(data, centers, 'euclidean')
        change_flag = False
        for i in range(N):
            j = np.argmin(distance[i, :])
            if assignments[i] != j:
                assignments[i] = j
                change_flag = True
          
        # Stop if cluster assignments did not change
        if change_flag is False:
            return assignments

        # 3. Compute new center of each cluster (mean of the points)
        for j in range(k):
            centers[j] = np.mean(data[assignments == j], axis=0)
                
    return assignments


def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find cdist (imported from scipy.spatial.distance) and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        pre_assignments = assignments.copy()
        
        # 2. Assign each point to the closest center
        distance = cdist(features, centers, 'euclidean')
        assignments = np.argmin(distance, axis=1)
          
        # Stop if cluster assignments did not change
        if np.all(pre_assignments == assignments):
            break
   
        # 3. Compute new center of each cluster (mean of the points)
        for j in range(k):
            centers[j] = np.mean(features[assignments == j], axis=0)

    return assignments



def hierarchical_clustering_using_complete_linkage(data, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.
    
    Hints
    - You may find pdist (imported from scipy.spatial.distance) useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = data.shape
    assert N >= k, 'Number of clusters cannot be greater than number of points'

    n_clusters = N # initially N clusters

    # Assign each point to its own cluster
    assignments = np.arange(N, dtype=np.uint32)

    # pdist: compute distance between all pairs of clusters, return condensed vector
    # squareform: transform the condensed vector to square form
    # distance matrix is a symmetric matrix
    distance_matrix = squareform(pdist(data, 'euclidean'))
    # distance_matrix[i,i] = 0, fill it with np.inf to avoid overflow and hardcode finite number like 1e9
    np.fill_diagonal(distance_matrix, np.inf)

    while n_clusters > k:

        # get two pairs that has smallest distance (i, j) = (row, col)
        i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)

        # ensure i is always the smaller index
        if j < i:
            i, j = j, i

        # the closest pair is i-th cluster and j-th cluster
        # merge the closest pair, e.g. merge i-th cluster to j-th cluster
        assignments[assignments == j] = i

        # decrement cluster labels for all points in clusters with labels higher than 'col'
        # this maintains consecutive cluster labeling after removing the 'col' cluster
        assignments[assignments > j] -= 1

        # update distance matrix: recomputing distances for the new merged cluster.
        # cluster distance - complete linkage: distance of two farthest members in each cluster
        # In this case, complete linkage performs better than single linkage, we use it for simplicity.
        # however, sklearn.clusters.AgglomerativeClustering uses ward's linkage, even better result than complete linkage
        distance_matrix[i, :] = np.maximum(distance_matrix[i, :], distance_matrix[j, :])
        distance_matrix[:, i] = distance_matrix[i, :]

        # remove the row and column corresponding to cluster j
        mask = np.ones(distance_matrix.shape[0], dtype=bool)
        mask[j] = False
        distance_matrix = distance_matrix[mask][:, mask]
        # print(f"remove the row and column corresponding to cluster {j}")
        # print(distance_matrix)

        # ensure diagonal is inf
        np.fill_diagonal(distance_matrix, np.inf)
        n_clusters -= 1

    return assignments


def hierarchical_clustering_using_centroid_linkage(data, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Hints
    - You may find pdist (imported from scipy.spatial.distance) useful

    Args:
        data - Each row represents a feature vector. N rows of data.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """


    N, D = data.shape
    assert N >= k, 'Number of clusters cannot be greater than number of points'

    n_clusters = N

    # Assign each point to its own cluster
    assignments = np.arange(N, dtype=np.uint32)
    centroids = np.copy(data)

    # distance matrix: distance between each pair of centroids
    distance_matrix = squareform(pdist(centroids, 'euclidean'))
    np.fill_diagonal(distance_matrix, np.inf)

    while n_clusters > k:
        # find the pair of cluster (i, j) that has smallest distance
        i, j = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
        if j < i:
            i, j = j, i

        # merge cluster i and cluster j
        assignments[assignments == j] = i
        assignments[assignments > j] -= 1

        # update centroid for the new cluster i
        centroids[i] = np.mean(data[assignments == i], axis=0)

        # remove cluster j from centroids
        mask = np.ones(n_clusters, dtype=bool)
        mask[j] = False
        centroids = centroids[mask]
        # remove from distance matrix corresponds to cluster j
        distance_matrix = distance_matrix[mask][:, mask]

        # update distance matrix with new centroid
        new_dist = np.linalg.norm(centroids - centroids[i], axis=1)
        distance_matrix[i, :] = new_dist
        distance_matrix[:, i] = new_dist

        # ensure diagonal is inf
        np.fill_diagonal(distance_matrix, np.inf)

        n_clusters -= 1

    return assignments

### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)

#    features = np.zeros((H*W, C))
#    for i in range(H):
#        for j in range(W):
#            # features[i] = features[i,:],  img[i,j] = img[i,j,:] = (r,g,b) in img[i,j]
#            features[i*W+j] = img[i,j]
#
#    however, this for loop doesn't use the power of numpy array

    features = img.reshape(H*W, C)
            
    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    img = img_as_float(img)

    # features = np.zeros((H*W, C+2))
    # for i in range(H):
    #     for j in range(W):
    #         features[i*W+j] = np.concatenate((img[i,j], np.array([i,j])))
    #
    #   however, this for loop doesn't use the power of numpy array

    # list all coordinates of image
    rows, cols = np.mgrid[0:H:1, 0:W:1]       # shape (H, W)
    features = np.dstack((img, rows, cols))   # shape (H, W, C+2)
    features = features.reshape((H*W, C+ 2))

    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, 3*C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    Iy = np.gradient(img, axis=0)        # shape (H, W, C)
    Ix = np.gradient(img, axis=1)        # shape (H, W, C)
    features = np.dstack((img, Ix, Iy))  # shape (H, W, 3*C)
    features = features.reshape((H*W, 3*C))
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    
    return features


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    # H, W = mask.shape
    # np.mean(mask_gt == mask) equals to np.sum(mask_gt == mask) / (H*W)
    accuracy = np.mean(mask_gt == mask)
    
    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
