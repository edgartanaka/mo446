import numpy as np


def pinv(A):
    '''
    https://hadrienj.github.io/posts/Deep-Learning-Book-Series-2.9-The-Moore-Penrose-Pseudoinverse/
    '''
    U, D, V = np.linalg.svd(A)
    D_plus = np.zeros((A.shape[0], A.shape[1])).T
    under_D = 1 / D  # this is the inverse of D, because D is a diagonal matrix
    D_inv = np.zeros((under_D.shape[0], under_D.shape[0]))
    row, col = np.diag_indices(D_inv.shape[0])
    D_inv[row, col] = under_D
    D_plus[:D.shape[0], :D.shape[0]] = D_inv
    A_plus = V.T.dot(D_plus).dot(U.T)

    return A_plus


def get_affine(original, transformed):
    # checking dimensions
    assert (original.shape[1] == 2)
    assert (transformed.shape[1] == 2)
    assert (transformed.shape[0] == original.shape[0])

    # building the X matrix
    matches_count = original.shape[0]
    zeros = np.zeros((matches_count, 3))
    coordinates_with_1 = np.append(original, np.ones((matches_count, 1)), axis=1)
    x_left = np.column_stack((coordinates_with_1, zeros)).reshape(2 * matches_count, 3)
    x_right = np.column_stack((zeros, coordinates_with_1)).reshape(2 * matches_count, 3)
    x = np.column_stack((x_left, x_right)).astype(int)

    # building the Y matrix
    y = transformed.reshape((2 * matches_count, 1))

    return x, pinv(x.T.dot(x)).dot(x.T.dot(y)), y


def comp_points(model):
    a, b, c, d, e, f = model.flatten().tolist()

    def comp_(point):
        x_ = point[0] * a + point[1] * b + c
        y_ = point[0] * d + point[1] * e + f

        return x_, y_

    return comp_


def comp_error(original, transformed, model):
    transformation = comp_points(model)
    B_fit = np.apply_along_axis(transformation, axis=1, arr=original)

    return np.sum((transformed - B_fit) ** 2, axis=1)


def RANSAC(original, transformed, n, k, t, d, return_error=False):
    """
    Implementation from pseudocode at
    https://en.wikipedia.org/wiki/Random_sample_consensus
    
    :param original: array (M x 2) containing the coordinates in the original image - one coordinate [x,y] per row
    :param transformed: array (M x 2) containing the coordinates in the transformed image - one coordinate [x,y] per row
    :param n:        the minimum number of data values required to fit the model
    :param k:        the maximum number of iterations allowed in the algorithm
    :param t:        a threshold value for determining when a data point fits a model
    :param d:        the number of close data values required to assert that a model fits well to data
    """
    best_inliers_count = 0
    best_inliers_idx = None
    errors = []
    indexes = np.arange(original.shape[0])

    for it in range(k):
        np.random.shuffle(indexes)

        chosen_original = original[indexes[:n]]  # maybeinliers
        chosen_transformed = transformed[indexes[:n]]

        x, maybemodel, y = get_affine(chosen_original, chosen_transformed)
        test_original = original[indexes[n:]]  # rest of the original data
        test_transformed = transformed[indexes[n:]]  # rest of the original transformed

        error = comp_error(test_original, test_transformed, maybemodel)
        errors.append(error)
        altoinsiers_idx = indexes[n:][error < t]

        alsoinliers = original[altoinsiers_idx]
        inliers_count = len(alsoinliers)
        # print('RANSAC - inliers_count:', inliers_count)
        if inliers_count > d and inliers_count > best_inliers_count:
            best_inliers_idx = altoinsiers_idx

    # print('best_inliers_idx:', best_inliers_idx)
    # print('original shape:', original.shape)

    if return_error:
        return original[best_inliers_idx], transformed[best_inliers_idx], errors
    return original[best_inliers_idx], transformed[best_inliers_idx]


def get_motion_affine_transformation(original, transformed):
    # Search for the best 3-point affine transformation and get the points that matched
    original, transformed = RANSAC(original, transformed, 3, 100, 9000, 2)

    # Find affine transformation on all of the matched points
    affine_transform = get_affine(original, transformed)

    return affine_transform
