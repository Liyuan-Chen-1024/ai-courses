"""
CS131 - Computer Vision: Foundations and Applications
Assignment 1
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 07/2017
Last modified: 10/16/2017
Python Version: 3.5+
"""

import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    for m in range(Hi):
        for n in range(Wi):
            for i in range(int(-Hk/2), int(Hk/2) + 1):
                for j in range(int(-Wk/2), int(Wk/2) + 1):
                    k_x, k_y = i + int(Hk/2), j + int(Wk/2) 
                    img_x, img_y = m - i, n - j
                    
                    if 0 <= img_x < Hi and 0 <= img_y < Wi:
                        out[m, n] += kernel[k_x, k_y] * image[img_x, img_y]

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H+2*pad_height, W+2*pad_width))
    
    out[pad_height:H+pad_height, pad_width:W+pad_width] = image

    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    v_kernel = np.flip(kernel, axis=0) # flip vertically
    folded_kernel = np.flip(v_kernel, axis=1) # flip horizontally
    
    padded_image = zero_pad(image, int(Hk/2), int(Wk/2))

    for i in range(Hi):
        for j in range(Wi):   
            # origin image => i - int(Hk/2): i + int(Hk/2) + 1
            # padded image => origin image + padding_height => i: i + Hk
            out[i, j] = np.sum(folded_kernel*padded_image[i:i+Hk, j:j+Wk])
                               
    return out

                               
def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    
    # flip g, then flip g again in the conv_fast() => kernel doesn't flip
    g = np.flip(g, axis=0) # flip vertically
    g = np.flip(g, axis=1) # flip horizontally
    
    out = conv_fast(f, g)

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
    gm = g - np.mean(g)
    out = cross_correlation(f, gm)

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """
  
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    
    padded_f = zero_pad(f, int(Hg/2), int(Wg/2))

    normalized_g = (g - np.mean(g))/np.std(g)

    for i in range(Hf):
        for j in range(Wf):   
            sub_f = padded_f[i:i+Hg, j:j+Wg]
            normalized_sub_f = (sub_f - np.mean(sub_f))/np.std(sub_f)
            out[i, j] = np.sum(normalized_g*normalized_sub_f)
            
    return out
                               
