import numpy as np
from scipy.ndimage import affine_transform

# Functions to convert points to homogeneous coordinates and back
pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
unpad = lambda x: x[:,:-1]

def plot_matches(ax, image1, image2, keypoints1, keypoints2, matches,
                 keypoints_color='k', matches_color=None, only_matches=False):
    """Plot matched features.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matches and image are drawn in this ax.
    image1 : (N, M [, 3]) array
        First grayscale or color image.
    image2 : (N, M [, 3]) array
        Second grayscale or color image.
    keypoints1 : (K1, 2) array
        First keypoint coordinates as ``(row, col)``.  ===> NOTE: row -> y, col -> x
    keypoints2 : (K2, 2) array
        Second keypoint coordinates as ``(row, col)``.
    matches : (Q, 2) array
        Indices of corresponding matches in first and second set of
        descriptors, where ``matches[:, 0]`` denote the indices in the first
        and ``matches[:, 1]`` the indices in the second set of descriptors.
    keypoints_color : matplotlib color, optional
        Color for keypoint locations.
    matches_color : matplotlib color, optional
        Color for lines which connect keypoint matches. By default the
        color is chosen randomly.
    only_matches : bool, optional
        Whether to only plot matches and not plot the keypoint locations.
    """

    image1.astype(np.float32)
    image2.astype(np.float32)

    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    image = np.concatenate([image1, image2], axis=1)

    offset = image1.shape

    if not only_matches:
        ax.scatter(keypoints1[:, 1], keypoints1[:, 0],
                   facecolors='none', edgecolors=keypoints_color)
        ax.scatter(keypoints2[:, 1] + offset[1], keypoints2[:, 0],
                   facecolors='none', edgecolors=keypoints_color)

    ax.imshow(image, interpolation='nearest', cmap='gray')
    ax.axis((0, 2 * offset[1], offset[0], 0))

    for i in range(matches.shape[0]):
        idx1 = matches[i, 0]  # idx1 in keypoint1
        idx2 = matches[i, 1]  # matched idx2 in keypoint2

        if matches_color is None:
            color = np.random.rand(3)  # color = (r,g,b)
        else:
            color = matches_color

        ax.plot((keypoints1[idx1, 1], keypoints2[idx2, 1] + offset[1]), # (x1, x2)
                (keypoints1[idx1, 0], keypoints2[idx2, 0]),             # (y1, y2)
                '-', color=color)


def get_output_space(img_ref, imgs, transforms):
    """
    Args:
        img_ref: reference image
        imgs: images to be transformed
        transforms: list of affine transformation matrices. transforms[i] maps
            points in imgs[i] to the points in img_ref
    Returns:
        output_shape
    """
    
    assert (len(imgs) == len(transforms))

    r, c = img_ref.shape
    corners = np.array([[0, 0], [r, 0], [0, c], [r, c]])
    all_corners = [corners]

    for i in range(len(imgs)):
        r, c = imgs[i].shape
        H = transforms[i]
        corners = np.array([[0, 0], [r, 0], [0, c], [r, c]])
        # y = A*p, if A = [[B 1]], then p = [[m], 
        #                                    [c]], for line y = m*x + c
        
        # here 'x' is input (x1,y1) map to 'y' output(x2, y2)
        # m = H[:2,:2], c = H[2,:2]
        warped_corners = corners.dot(H[:2,:2]) + H[2,:2]
        all_corners.append(warped_corners)

    # Find the extents of both the reference image and the warped
    # target image
    all_corners = np.vstack(all_corners)

    # The overall output shape will be max - min
    corner_min = np.amin(all_corners, axis=0)
    corner_max = np.amax(all_corners, axis=0)
    output_shape = (corner_max - corner_min)

    # Ensure integer shape with np.ceil and dtype conversion
    output_shape = np.ceil(output_shape).astype(int)
    offset = corner_min

    return output_shape, offset

def warp_image(img, H, output_shape, offset):

    # Note about affine_transfomr function:
    # Given an output image pixel index vector o,
    # the pixel value is determined from the input image at position
    # np.dot(matrix, o) + offset.
    # If you have a matrix for the ‘push’ transformation, transforming input (img) to output (img * H = p1). Use its inverse (numpy.linalg.inv) in this function.
    # H = [[a,b,0],          Hinv = [[o,p,q],
    #      [c,d,0],   ==>            [m,n,s],    
    #      [e,f,1]]                  [0,0,1]]
    # H = [[m]        ==>    Hinv = [[m, b]]
    #      [c]]
    Hinv = np.linalg.inv(H)
    m = Hinv.T[:2,:2]
    b = Hinv.T[:2,2] 
    # offset = b + offset ???
    img_warped = affine_transform(img.astype(np.float32),
                                  m, 
                                  b+offset,
                                  output_shape,
                                  cval=-1)

    return img_warped

