"""
CS131 - Computer Vision: Foundations and Applications
Assignment 2
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/18/2017
Python Version: 3.5+
"""

import numpy as np
from collections import deque

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2  # axis = 0
    pad_width1 = Wk // 2  # axis = 1
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')
    
    # flip kernel
    kernel = np.flip(np.flip(kernel, 0), 1)
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(kernel*padded[i:i + Hk, j:j+Wk])

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """
    
    kernel = np.zeros((size, size))
    k = (size - 1) // 2
    
    for i in range(size):
        for j in range(size):
            kernel[i,j] = 1/(2*np.pi*sigma**2)*np.exp(-1*((i - k)**2 + (j-k)**2)/(2*sigma**2))

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    Dx = 1.0/2*np.array([[1,0,-1]])
    # Sx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    out = conv(img, Dx)
    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    Dy = 1.0/2*np.array([[1],[0],[-1]])
    # Sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    out = conv(img, Dy)
    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    fx = partial_x(img)
    fy = partial_y(img)
    
    G = np.sqrt(fx**2 + fy**2)
    theta = (np.rad2deg(np.arctan2(fy, fx)) + 180) % 360

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    
    for i in range(1, H-1):
        for j in range(1, W-1):
            alpha=np.deg2rad(theta[i,j])
            p1 = G[i - int(np.round(np.sin(alpha))), j - int(np.round(np.cos(alpha)))]
            p2 = G[i + int(np.round(np.sin(alpha))), j + int(np.round(np.cos(alpha)))]
            if G[i, j] > p1 and G[i, j] > p2:
                out[i, j] = G[i,j]
            else:
                out[i,j] = 0
    
    return out


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=bool)
    weak_edges = np.zeros(img.shape, dtype=bool)

    strong_edges = img > high
    weak_edges = np.logical_and(img <= high, img > low)

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype='bool')

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    # BFS
    queue = deque(indices)
    while queue:
        cy, cx = queue.popleft()
        neighbors = get_neighbors(cy,cx,H,W)
        for ny, nx in neighbors:
            if weak_edges[ny, nx] != 0:
                edges[ny, nx] = weak_edges[ny, nx]
                queue.append([ny, nx])
                weak_edges[ny, nx] = 0
                
    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    # 1. smoothing
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed_img = conv(img, kernel)
    
    # 2. finding gradients
    G, theta = gradient(smoothed_img)
    
    # 3. non-maximum suppresion
    suppressed_img = non_maximum_suppression(G, theta)
    
    # 4. threshold
    strong_edges, weak_edges = double_thresholding(suppressed_img, high, low)
    
    # 5. link edges
    edge = link_edges(strong_edges, weak_edges)
    
    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    H, W = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)  # rho value range: [-rho_max, rho_max]
    thetas = np.deg2rad(np.linspace(-90.0, 90.0, 180 + 1))                

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)

    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    for i in range(len(ys)):
        x, y = xs[i], ys[i]
        rho_t = x*cos_t + y*sin_t
        for i in range(len(rho_t)):
            rho_index = np.argmin(np.abs(rhos - rho_t[i]))
            accumulator[rho_index, i] += 1
    return accumulator, rhos, thetas
