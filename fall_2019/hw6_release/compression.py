import numpy as np


def compress_image(image, num_values):
    """Compress an image using SVD and keeping the top `num_values` singular values.

    Args:
        image: numpy array of shape (H, W)
        num_values: number of singular values to keep

    Returns:
        compressed_image: numpy array of shape (H, W) containing the compressed image
        compressed_size: size of the compressed image
    """
    compressed_image = np.zeros_like(image)
    compressed_size = 0

    # Steps:
    #     1. Get SVD of the image
    #     2. Only keep the top `num_values` singular values, and compute `compressed_image`
    #     3. Compute the compressed size
    
    # 1. Get SVD of the image
    U, S, Vt = np.linalg.svd(image)
    # sigma = np.zeros((image.shape))
    # np.fill_diagonal(sigma, S)
    # print(np.isclose(U @ sigma @ Vt, image)) # True,...
    
    # 2. Only keep the top `num_values` singular values, and compute `compressed_image`
    for i in range(num_values):
        u = np.expand_dims(U[:,i], axis=1)    # m * 1
        s = S[i]
        v = np.expand_dims(Vt[i,:], axis=0)  # 1 * n
        compressed_image += (u*s).dot(v)      # plus partial A
      
    # 3. Compute the compressed size
    compressed_size = (U.shape[0] + 1 + Vt.shape[0]) * num_values   # (m + 1 + n) * k
    

    assert compressed_image.shape == image.shape, \
           "Compressed image and original image don't have the same shape"

    assert compressed_size > 0, "Don't forget to compute compressed_size"

    return compressed_image, compressed_size
