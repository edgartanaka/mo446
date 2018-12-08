""""
University of Campinas

MO446 - Computer Vision (Prof. Adin)
Final Project - Image Memorability

Authors:
Darley Barreto
Edgar Tanaka
Tiago Barros

Scope of this file:
Computation of global features for target images.
"""

from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
from scipy.io import loadmat
from scipy.io import savemat
from tqdm import tqdm
import numpy as np
import gist  # https://github.com/tuttieee/lear-gist-python
import cv2
import os

# Images dimensions
IMG_HEIGHT = 256
IMG_WIDTH  = 256

# Distance between grid centers
GRID_SPACING = 1

# Size of patch from which to compute descriptors (must be a factor of 4)
PATCH_SIZE = 16

# Number of spatial windows for computing histograms
NUM_SPATIALSCALES = 2

# Number of visual words
NUM_VISUALWORDS = 200

# Number of samples of descriptors
NUM_SAMPLES = 20

data_path = "input/Data/"

# File containing the input target images
target_images = loadmat(os.path.join(data_path, "Image data/target_images.mat"))
target_images = target_images["img"]

num_images = target_images.shape[3]

# Grid for the dense feature extraction
grid_y = np.arange(PATCH_SIZE / 2,
                   IMG_HEIGHT - PATCH_SIZE / 2 + 2,
                   GRID_SPACING)
grid_x = np.arange(PATCH_SIZE / 2,
                   IMG_WIDTH - PATCH_SIZE / 2 + 2,
                   GRID_SPACING)
nrows = len(grid_y)
ncols = len(grid_x)

#cdata = loadmat(os.path.join(data_path, "Image data/target_features_ours2.mat"))


def main():
    items = {}

    items["pixel_histograms"] = createPixelHistograms()
#    hpixel = {"pixel_histograms" : items["pixel_histograms"]}
#    savemat(os.path.join(data_path, "Image data/target_features_pixel.mat"), hpixel)
    items["gist"] = computeGist()
#    hgist = {"gist" : items["gist"]}
#    savemat(os.path.join(data_path, "Image data/target_features_gist.mat"), hgist)
    items["sptHistsift"]  = sift()
#    hsift = {"sptHistsift" : items["sptHistsift"]}
#    savemat(os.path.join(data_path, "Image data/target_features_sift.mat"), hsift)
    items["sptHistsurf"]  = surf()
#    hsurf = {"sptHistsurf" : items["sptHistsurf"]}
#    savemat(os.path.join(data_path, "Image data/target_features_surf.mat"), hsurf)
    items["sptHistorb"]   = orb()
#    horb = {"sptHistorb" : items["sptHistorb"]}
#    savemat(os.path.join(data_path, "Image data/target_features_orb.mat"), horb)
    items["sptHistbrisk"] = brisk()
#    hbrisk = {"sptHistbrisk" : items["sptHistbrisk"]}
#    savemat(os.path.join(data_path, "Image data/target_features_brisk.mat"), hbrisk)
#    items["sptHisthog"]   = hog()
#    hhog = {"sptHisthog" : items["sptHisthog"]}
#    savemat(os.path.join(data_path, "Image data/target_features_hog.mat"), hhog)

    savemat(os.path.join("output", "target_features.mat"), items)


def sift() -> np.ndarray:
    print("\n=== SIFT ===")
    return computeHistograms("sift")


def surf() -> np.ndarray:
    print("\n=== SURF ===")
    return computeHistograms("surf")


def orb() -> np.ndarray:
    print("\n=== ORB ===")
    return computeHistograms("orb")


def brisk() -> np.ndarray:
    print("\n=== BRISK ===")
    return computeHistograms("brisk")


def hog() -> np.ndarray:
    print("\n=== HOG ===")
    return computeHistograms("hog")


def computeHistograms(descriptor) -> np.ndarray:
    """
    Computes the histograms of features to be used in the regressor.
    :param descriptor: Descriptor to be used.
    :return: Histograms (one per input image) of features.
    """
    if descriptor is None:
        return None

    if descriptor == "sift":
        num_features = 128
    elif descriptor == "surf":
        num_features = 64
    elif descriptor == "orb":
        num_features = 32
    elif descriptor == "brisk":
        num_features = 64
    elif descriptor == "hog":
        num_features = cv2.HOGDescriptor().getDescriptorSize()

    # Uses each 10th image to build the clusters of features
    images_with_steps = np.arange(0, num_images, 10)

    size = NUM_SAMPLES * len(images_with_steps)

    all_feature_vectors = np.zeros([size, num_features])

    k = 0

    print("Computing features..... (1/3)")
    for i in tqdm(images_with_steps):
        img = target_images[:, :, :, i]
        feature_vectors = extractFeatures(img, descriptor)

        # Performs a sampling of the descriptors
        n = feature_vectors.shape[0]
        r = np.random.permutation(n)
        r = r[0 : NUM_SAMPLES]
        all_feature_vectors[k : k + NUM_SAMPLES] = feature_vectors[r]
        k += NUM_SAMPLES

#    savemat(os.path.join(data_path, "Image data/target_features_ours0.mat"),
#            {"items": all_feature_vectors})

    print("Computing clusters..... (2/3)")
    kmeans = cluster(all_feature_vectors)

    centers = kmeans.cluster_centers_

#    savemat(os.path.join(data_path, "Image data/target_features_ours1.mat"),
#            {"items": centers})

    # Sorts the centers of the clusters using the first principal component
    pc = pca(all_feature_vectors.T, 2)
    pc1 = pc[:, 0] @ centers.T
    k = np.argsort(pc1)
    centers = centers[k]

#    savemat(os.path.join(data_path, "Image data/target_features_ours2.mat"),
#            {"items": centers})

#    centers = cdata["items"]

    # Output histogram of features
    sptHist = np.zeros([round(NUM_VISUALWORDS *
                              ((4 ** NUM_SPATIALSCALES - 1) / 3)),
                        num_images])

    print("Computing histograms... (3/3)")
    for i in tqdm(range(num_images)):
        img = target_images[:, :, :, i]
        feature_vectors = extractFeatures(img, descriptor)
        W = np.argmin(distMat(centers, feature_vectors), axis = 0)
        W = np.reshape(W, [nrows, ncols])
        sptHist[:, i] = spatialHistogram(W)

    return sptHist.T


def extractFeatures(img: np.ndarray, descriptor: str) -> np.ndarray:
    """
    Computes feature vectors using a descriptor.
    :param img: Input colored image.
    :param descriptor: Descriptor to use.
    :return: Dense feature vectors
    """
    if descriptor is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Builds the grid for dense keypoint description
    if descriptor == "hog":
        kps = [(x, y) for y in grid_y for x in grid_x]
    else:
        kps = [cv2.KeyPoint(x, y, GRID_SPACING) for y in grid_y for x in grid_x]

    # Builds the feature vectors
    if descriptor == "sift":
        sift = cv2.xfeatures2d.SIFT_create()
        [kps, descs] = sift.compute(gray, kps)
    elif descriptor == "surf":
        surf = cv2.xfeatures2d.SURF_create()
        [kps, descs] = surf.compute(gray, kps)
    elif descriptor == "orb":
        orb = cv2.ORB_create(edgeThreshold = 2, patchSize = PATCH_SIZE)
        [kps, descs] = orb.compute(gray, kps)
    elif descriptor == "brisk":
        brisk = cv2.BRISK_create(patternScale = 0.4)
        [kps, descs] = brisk.compute(gray, kps)
    elif descriptor == "hog":
        hog = cv2.HOGDescriptor()
        descs = hog.compute(gray, locations = kps)
        descs = np.reshape(descs, [len(kps), hog.getDescriptorSize()])

    return np.array(descs, dtype=np.float64)


def cluster(feature_vectors: np.ndarray) -> KMeans:
    """
    Computes the clusters of feature vectors using k-means.
    :param feature_vectors: All feature vectors to consider.
    :return: Instance of fitted k-means.
    """
    return KMeans(n_clusters = NUM_VISUALWORDS, init = "k-means++", n_init = 1,
                  max_iter = 800).fit(feature_vectors)


def pca(features: np.ndarray, k: int) -> np.ndarray:
    """
    Computes the principal components of the features.
    :param features: Features to consider.
    :param k: Number of eigenvectors to compute.
    """
    mu = np.mean(features, axis = 1).reshape(-1, 1)
    fm = features - np.tile(mu, [1, features.shape[1]])
    X = fm @ fm.T

    latent, pc = eigs(X, k)

    return pc


def distMat(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """
    Computes the distance between feature vectors.
    :param P1: Feature vector.
    :param P2: Feature vector.
    :return: Distance
    """
    n1 = P1.shape[0]
    n2 = P2.shape[0]

    norm1 = np.sum(P1 ** 2, 1).reshape(-1, 1)
    norm2 = np.sum(P2 ** 2, 1).reshape(-1, 1)

    X1 = np.tile(norm1, [1, n2])
    X2 = np.tile(norm2, [1, n1])

    R = P1 @ P2.T
    D = X1 + X2.T - 2 * R

    return D


def spatialHistogram(W: np.ndarray) -> np.ndarray:
    """
    Computes the histogram of features for one image.
    :param W: Array of indices of features that minimise distance.
    :return: Histogram of features.
    """
    c1 = 2 ** (NUM_SPATIALSCALES - 1)
    c2 = 2 ** (NUM_SPATIALSCALES - (np.arange(1, NUM_SPATIALSCALES)))
    coef = 1 / np.append(c1, c2)

    h = []

    for M in range(NUM_SPATIALSCALES):
        lx = np.around(np.linspace(0,
                                   W.shape[1] - 2,
                                   2 ** M + 1)).astype(int)
        ly = np.around(np.linspace(0,
                                   W.shape[0] - 2,
                                   2 ** M + 1)).astype(int)

        for x in range(2 ** M):
            for y in range(2 ** M):
                ww = W[ly[y] : ly[y + 1], lx[x] : lx[x + 1]]
                hh = np.bincount(ww.ravel(), minlength = NUM_VISUALWORDS)
                h = np.append(h, hh * coef[M])

    h = h / np.sum(h)

    return h


def createPixelHistograms() -> np.ndarray:
    """
    Computes histograms of pixels, which are used as feature vectors.
    """
    print("\n=== Pixel Histograms ===")

    nbins = 21
    bins = np.linspace(0, 1.05, nbins + 1) - 0.025
    pixel_histograms = np.zeros([num_images, nbins, 3])

    print("Computing histograms... (1/1)")
    for i in tqdm(range(num_images)):
        img = target_images[:, :, :, i]

        rs = img[:, :, 0] / 255
        gs = img[:, :, 1] / 255
        bs = img[:, :, 2] / 255

        pixel_histograms[i, :, 0] = np.histogram(rs.ravel(), bins)[0]
        pixel_histograms[i, :, 1] = np.histogram(gs.ravel(), bins)[0]
        pixel_histograms[i, :, 2] = np.histogram(bs.ravel(), bins)[0]

    return pixel_histograms


def computeGist() -> np.ndarray:
    """
    Computes GIST features for all images.
    """
    print("\n=== GIST ===")

    num_features = 1536
    all_feature_vectors = np.zeros([num_images, num_features])

    print("Computing gist... (1/1)")
    for i in tqdm(range(num_images)):
        img = target_images[:, :, :, i]

        feature_vectors = gist.extract(img, 4, [8, 8, 8, 8])
        all_feature_vectors[i] = feature_vectors

    return all_feature_vectors


if __name__ == "__main__":
    main()
