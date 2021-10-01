"""
CS131 - Computer Vision: Foundations and Applications
Assignment 3
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 09/27/2018
Python Version: 3.5+
"""

import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from skimage.util.shape import view_as_blocks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve, correlate

from utils import pad, unpad, get_output_space, warp_image


def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve,
        which is already imported above. If you use convolve(), remember to
        specify zero-padding to match our equations, for example:

            out_image = convolve(in_image, kernel, mode='constant', cval=0)

        You can also use for nested loops compute M and the subsequent Harris
        corner response for each output pixel, intead of using convolve().
        Your implementation of conv_fast or conv_nested in HW1 may be a
        useful reference!

    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    # 1. Compute x and y derivatives (I_x, I_y) of an image
    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    A = convolve(dx**2, window)
    B = convolve(dy**2, window)
    C = convolve(dx*dy, window)
    
    det_M = A*B - C**2
    trace_M = A + B
    
    response = det_M - k*trace_M**2
    
    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard
    normal distribution (having mean of 0 and standard deviation of 1)
    and then flattening into a 1D array.

    The normalization will make the descriptor more robust to change
    in lighting condition.

    Hint:
        In this case of normalization, if a denominator is zero, divide by 1 instead.

    Args:
        patch: grayscale image patch of shape (H, W)

    Returns:
        feature: 1D array of shape (H * W)
    """
    H, W = patch.shape
    
    normalized_path = (patch - np.mean(patch))/np.std(patch)
    
    feature = normalized_path.flatten()
    
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint

    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed
    when the distance to the closest vector is much smaller than the distance to the
    second-closest, that is, the ratio of the distances should be strictly smaller
    than the threshold (not equal to). Return the matches as pairs of vector indices.

    Hint:
        The Numpy functions np.sort, np.argmin, np.asarray might be useful

        The Scipy function cdist calculates Euclidean distance between all
        pairs of inputs
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints

    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair
        of matching descriptors
    """
    matches = []

    M = desc1.shape[0]
    dists = cdist(desc1, desc2)

    for i in range(M):
        dist_i = dists[i]
        min_idx = np.argmin(dist_i)
        sorted_dist_i = np.sort(dist_i)
        if sorted_dist_i[0] / sorted_dist_i[1] < threshold:
            matches.append((i, min_idx))
            
    matches = np.asarray(matches)
        
    return matches


def fit_affine_matrix(p1, p2):
    """ 
    Fit affine matrix such that p2 * H = p1. First, pad the descriptor vectors
    with a 1 using pad() to convert to homogeneous coordinates, then return
    the least squares fit affine matrix in homogeneous coordinates.

    Hint:
        You can use np.linalg.lstsq function to solve the problem. 

        Explicitly specify np.linalg.lstsq's new default parameter rcond=None 
        to suppress deprecation warnings, and match the autograder.

    Args:
        p1: an array of shape (M, P) holding descriptors of size P about M keypoints
        p2: an array of shape (M, P) holding descriptors of size P about M keypoints

    Return:
        H: a matrix of shape (P+1, P+1) that transforms p2 to p1 in homogeneous
        coordinates
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    H = np.linalg.lstsq(p2, p1, rcond=None)[0]
    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation:

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers via Euclidean distance
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Update max_inliers as a boolean array where True represents the keypoint
    at this index is an inlier, while False represents that it is not an inlier.

    Hint:
        You can use np.linalg.lstsq function to solve the problem. 

        Explicitly specify np.linalg.lstsq's new default parameter rcond=None 
        to suppress deprecation warnings, and match the autograder.

        You can compute elementwise boolean operations between two numpy arrays,
        and use boolean arrays to select array elements by index:
        https://numpy.org/doc/stable/reference/arrays.indexing.html#boolean-array-indexing 

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
        robust_matches:
    """
    # Copy matches array, to avoid overwriting it
    orig_matches = matches.copy()
    matches = matches.copy()

    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = keypoints1[matches[:,0]]
    matched2 = keypoints2[matches[:,1]]
    
    p_matched1 = pad(matched1)
    p_matched2 = pad(matched2)

    max_inliers = np.zeros(N, dtype=bool)
    n_inliers = 0   
    H = None

    # RANSAC iteration start
    
    # Note: while there're many ways to do random sampling, please use
    # `np.random.shuffle()` followed by slicing out the first `n_samples`
    # matches here in order to align with the auto-grader.
    # Sample with this code:
    '''
        np.random.shuffle(matches)
        samples = matches[:n_samples]
        sample1 = pad(keypoints1[samples[:,0]])
        sample2 = pad(keypoints2[samples[:,1]])
    '''
    
    for i in range(n_iters):
        # 1. Select random set of matches
        idx = np.random.choice(N, n_samples, replace=False)
        samples = matches[idx]
        sample1 = keypoints1[samples[:,0]]
        sample2 = keypoints2[samples[:,1]]
        
        # 2. Compute affine transformation matrix (use samples to get H)
        tmp_H = fit_affine_matrix(sample1, sample2)
        
        # 3. Compute inliers via Euclidean distance
        # m, c = H[:2,:2], H[2,:2]
        # after fit, 'y = matched2 * m + c' or '[y 1] = [matched2 1].dot(H)'
         
        # get Euclidean distance among all matched points
        distance = np.linalg.norm(p_matched2.dot(tmp_H) - p_matched1, axis=1)
        
        is_inliers = distance < threshold
       
        tmp_n_inliers = np.sum(is_inliers)
           
        # 4. Keep the largest set of inliers
        if tmp_n_inliers > n_inliers:
            n_inliers = tmp_n_inliers
            H = tmp_H
            max_inliers = is_inliers.copy() 
            
    # 5. Re-compute least-squares estimate on all of the inliers
    samples = matches[max_inliers]
    sample1 = keypoints1[samples[:,0]]
    sample2 = keypoints2[samples[:,1]]
    H = fit_affine_matrix(sample1, sample2)
        
    return H, orig_matches[max_inliers]


def hog_descriptor(patch, pixels_per_cell=(8,8)):
    """
    Generating hog descriptor by the following steps:

    1. Compute the gradient image in x and y directions (already done for you)
    2. Compute gradient histograms for each cell
    3. Flatten block of histograms into a 1D feature vector
        Here, we treat the entire patch of histograms as our block
    4. Normalize flattened block by L2 norm
        Normalization makes the descriptor more robust to lighting variations

    Args:
        patch: grayscale image patch of shape (H, W)
        pixels_per_cell: size of a cell with shape (M, N)

    Returns:
        block: 1D patch descriptor array of shape ((H*W*n_bins)/(M*N))
    """
    assert (patch.shape[0] % pixels_per_cell[0] == 0),\
                'Heights of patch and cell do not match'
    assert (patch.shape[1] % pixels_per_cell[1] == 0),\
                'Widths of patch and cell do not match'

    n_bins = 9
    degrees_per_bin = 180 // n_bins

    Gx = filters.sobel_v(patch)
    Gy = filters.sobel_h(patch)

    # Unsigned gradients
    G = np.sqrt(Gx**2 + Gy**2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi) % 180

    # Group entries of G and theta into cells of shape pixels_per_cell, (M, N)
    #   G_cells.shape = theta_cells.shape = (H//M, W//N)
    #   G_cells[0, 0].shape = theta_cells[0, 0].shape = (M, N)
    G_cells = view_as_blocks(G, block_shape=pixels_per_cell)
    theta_cells = view_as_blocks(theta, block_shape=pixels_per_cell)
    rows = G_cells.shape[0]
    cols = G_cells.shape[1]

    # For each cell, keep track of gradient histrogram of size n_bins
    cells = np.zeros((rows, cols, n_bins))

    # 2. Compute histogram per cell
    for i in range(rows):
        for j in range(cols):
            for m in range(pixels_per_cell[0]):
                for n in range(pixels_per_cell[1]):
                    t = theta_cells[i,j,m,n]
                    b = np.ceil((t - degrees_per_bin/2)/degrees_per_bin).astype(int)
                    b %= n_bins
                    cells[i,j,b] += G_cells[i,j,m,n]
        
    # 3. Flatten block of histograms into a 1D feature vector. 
    # Here, we treat the entire patch of histograms as our block
    block = cells.flatten()
    
    # 4. Normalize flattened block by L2 norm
    l2_norm = np.linalg.norm(block)
    block = block / l2_norm
#     for i in range(0, len(block), n_bins):
#         block[i:i+n_bins] = block[i:i+n_bins] / np.linalg.norm(block[i:i+n_bins])
    
    return block


def linear_blend(img1_warped, img2_warped):
    """
    Linearly blend img1_warped and img2_warped by following the steps:

    1. Define left and right margins (already done for you)
    2. Define a weight matrices for img1_warped and img2_warped
        np.linspace and np.tile functions will be useful
    3. Apply the weight matrices to their corresponding images
    4. Combine the images

    Args:
        img1_warped: Refernce image warped into output space
        img2_warped: Transformed image warped into output space

    Returns:
        merged: Merged image in output space
    """
    out_H, out_W = img1_warped.shape # Height and width of output space
    img1_mask = (img1_warped != 0)  # Mask == 1 inside the image
    img2_mask = (img2_warped != 0)  # Mask == 1 inside the image

    # 1. Define left and right margins for blending to occur between
    # Find column of middle row where warped image 1 ends
    # This is where to end weight mask for warped image 1
    # img1_mask: True:1, False:0.  
    # np.argmax(np.fliplr(img1_mask...)), can found the most right margin (=1) (flipped)
    # out_W - np.argmax(np.fliplr(..)), can return width from left to right where img1 ends
    right_margin = out_W - np.argmax(np.fliplr(img1_mask)[out_H//2, :].reshape(1, out_W), 1)[0]

    # Find column of middle row where warped image 2 starts
    # This is where to start weight mask for warped image 2
    left_margin = np.argmax(img2_mask[out_H//2, :].reshape(1, out_W), 1)[0]


    # 2. Define a weight matrix for image 1 such that: - From the left of the output space to the left margin the weight is 1 - From the left margin to the right margin, the weight linearly decrements from 1 to 0 
    w1 = np.zeros(out_W)
    w1[:left_margin] = 1
    w1[left_margin:right_margin] = np.linspace(1,0,right_margin-left_margin)
    w1 = np.tile(w1, (out_H, 1))
    
    # 3. Define a weight matrix for image 2 such that: - From the right of the output space to the right margin the weight is 1 - From the left margin to the right margin, the weight linearly increments from 0 to 1 
    w2 = np.zeros(out_W)
    w2[right_margin:] = 1
    w2[left_margin:right_margin] = np.linspace(0,1,right_margin-left_margin)
    w2 = np.tile(w2, (out_H, 1))
    
    # 4. Apply the weight matrices to their corresponding images 
    w_img1 = img1_warped * w1
    w_img2 = img2_warped * w2
    
    # 5. Combine the images
    merged = w_img1 + w_img2

    return merged


def stitch_multiple_images(imgs, desc_func=simple_descriptor, patch_size=5):
    """
    Stitch an ordered chain of images together.

    Args:
        imgs: List of length m containing the ordered chain of m images
        desc_func: Function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: Size of square patch at each keypoint

    Returns:
        panorama: Final panorma image in coordinate frame of reference image
    """
    # 1. Detect keypoints in each image
    keypoints = []  # keypoints[i] corresponds to imgs[i]
    for img in imgs:
        kypnts = corner_peaks(harris_corners(img, window_size=3),
                              threshold_rel=0.05,
                              exclude_border=8)
        keypoints.append(kypnts)
        
    # 2. Describe keypoints
    descriptors = []  # descriptors[i] corresponds to keypoints[i]
    for i, kypnts in enumerate(keypoints):
        desc = describe_keypoints(imgs[i], kypnts,
                                  desc_func=desc_func,
                                  patch_size=patch_size)
        descriptors.append(desc)
        
    # 3. Match keypoints in neighboring images
    matches = []  # matches[i] corresponds to matches between
                  # descriptors[i] and descriptors[i+1]
    for i in range(len(imgs)-1):
        mtchs = match_descriptors(descriptors[i], descriptors[i+1], 0.7)
        matches.append(mtchs)

    # 4. transforms[i] maps points in imgs[i] to the points in img_ref
    # p2*H1=p1, p3*H2 = p2 ==> p3*H2*H1 = p2*H1 = p1
   
    Hs = [np.eye(3)] # Hs[0] later for warp img1
    for i in range(len(imgs)-1):
        # append directly, to avoid shallow copy in array
        Hs.append(ransac(keypoints[i], keypoints[i+1], matches[i], threshold=1)[0])
        
    for i in range(1, len(imgs)):
        # Hs[i]*Hs[i-1] is wrong, use matmul instead
        Hs[i] = np.matmul(Hs[i], Hs[i - 1])
        
            
    # 5. get output shape
    output_shape, offset = get_output_space(imgs[0], imgs[1:], Hs[1:])
    
    # 6. Warp images into output space
    warped_imgs = []
    for i in range(len(imgs)):
        warped_imgs.append(warp_image(imgs[i], Hs[i], output_shape, offset))
        img_mask = (warped_imgs[-1] != -1)   # Mask == 1 inside the image
        warped_imgs[-1][~img_mask] = 0       # Return background values to 0
        
    # 7. linear_blend
    merged = warped_imgs[0]
    for i in range(1, len(imgs)):
        merged = linear_blend(merged, warped_imgs[i])
        
    panorama = merged

    return panorama

