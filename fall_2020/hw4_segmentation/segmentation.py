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
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

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

    # 1. Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        
        # 2. Assign each point to the closest center
        distance = cdist(centers, features, 'euclidean')
        change_flag = False
        for i in range(N):
            j = np.argmin(distance[:, i])
            if assignments[i] != j:
                assignments[i] = j
                change_flag = True
          
        # Stop if cluster assignments did not change
        if change_flag is False:
            return assignments
   
        # 3. Compute new center of each cluster (mean of the points)
        for j in range(k):
            centers[j] = np.mean(features[assignments == j])
                
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
        distance = cdist(centers, features, 'euclidean')
        assignments = np.argmin(distance, axis=0)
          
        # Stop if cluster assignments did not change
        if np.all(pre_assignments == assignments):
            break
   
        # 3. Compute new center of each cluster (mean of the points)
        for j in range(k):
            centers[j] = np.mean(features[assignments == j])

    return assignments



def hierarchical_clustering(features, k):
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
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """


    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N, dtype=np.uint32)
    centers = np.copy(features)
    n_clusters = N

    while n_clusters > k:
        # compute distance between all pairs of clusters
        distance = pdist(centers, 'euclidean')
        
        # sysmetric distance matrix
        matrix = squareform(distance)  
        matrix += np.eye(n_clusters)*1e9  # currently matrix[i,i] = 0, replace it with large number 
        
        # get the smallest distance, merge this pair of clusters
        idx = np.argmin(matrix)
       
        # m*n matrix, (i, j) -> idx: idx = m*i + j
        # idx -> (i, j), i = idx // m, j = idx - m*i
        row = idx // n_clusters   # n_cluster == distance.shape[0]
        col = idx - n_clusters * row
        
        # since it's a symmetric matrix, row and col can exchange
        # the closest pair is row-th cluster (row, col) and col-th cluster (col, row)
        # merge the closest pair, e.g. merge col-th cluster to row-th cluster
        for i in range(N):
            if assignments[i] == col:
                assignments[i] = row
        
        for i in range(N):
            if assignments[i] > col:
                assignments[i] -= 1  # delete col-th cluster from matrix, -1. prepare for next loop
                
        centers = np.delete(centers, col, axis=0)
        centers[row] = np.mean(features[assignments==row], axis=0)
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
    features = np.zeros((H*W, C))

#     for i in range(H):
#         for j in range(W):
#             # features[i] = features[i,:],  img[i,j] = img[i,j,:] = (r,g,b) in img[i,j]
#             features[i*W+j] = img[i,j] 
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
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    for i in range(H):
        for j in range(W):
            features[i*W+j] = np.concatenate((color[i,j], np.array([i,j])))
    
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
  
        
    features = color_features(img)
    
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
