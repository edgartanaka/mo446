""""
State University of Campinas

MO446 - Computer Vision (Prof. Adin)
Project 1 - Cartoonization

Group 4:
Darley Barreto
Edgar Tanaka
Tiago Barros

Scope of this file:
Cartoonize images - run the whole pipeline
"""

import glob
import cv2 as cv
import numpy as np
import os

NUMBER_OF_COLORS = 16
CANNY_LOW_THRESHOLD = 80
CANNY_HIGH_THRESHOLD = 150


def reduce_color_convolution(img):
    """
    Best attempt to reduce color using only convolutions.
    From all the experiments we did, the gaussian filter with 15x15 kernel was the one considered best.
    :param img:
    :return:
    """
    kernel = cv.getGaussianKernel(15, 2)
    return cv.filter2D(img, -1, kernel)


def reduce_color_proposed(img):
    """
    Improves the color reduction effect by applying k-means clusterization
    :param img:
    :return:
    """
    # First, we apply a median blur filter
    img_blur = cv.medianBlur(img, 7)

    # Reshaping the input image
    img_reshape = img_blur.reshape((-1, 3))
    img_reshape = np.float32(img_reshape)

    # Now we quantize the colors with k-means
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv.kmeans(img_reshape, NUMBER_OF_COLORS, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # Generates output image using the centers of clusters
    center = np.uint8(center)
    output = center[label.flatten()]
    output = output.reshape((img.shape))

    return output


def borders1(input_file):
    """
    Edge detection with convolutions.
    We are using the Laplacian kernel.
    :param input_file:
    :return:
    """
    # read as grayscale
    img = cv.imread(input_file, 0)

    # laplacian of gaussian centered on zero
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    edges = cv.filter2D(img, -1, kernel)
    edges = cv.adaptiveThreshold(edges, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    return edges


def borders2(reduced_img):
    """
    Canny edge detector on reduced image colored
    :param reduced_img:
    :return:
    """
    return cv.Canny(reduced_img, 80, 150)


def borders3(input_file):
    """
    Canny edge detector on original image colored
    :param input_file:
    :return:
    """

    # read as colored
    img = cv.imread(input_file, 1)
    return cv.Canny(img, 80, 150)


def borders4(input_file):
    """
    Canny edge detector on original image black and white
    :param input_file:
    :return:
    """

    # read as bw
    img = cv.imread(input_file, 0)
    return cv.Canny(img, 80, 180)


def borders5(input_file):
    """
    Canny edge detector on original image black and white + morphological transformation close
    :param input_file:
    :return:
    """
    img = cv.imread(input_file, 0)
    edges = cv.Canny(img, 80, 150)

    # morphological transformation closing (trying to make the border smoother)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    return edges


def borders6(input_file):
    """
    Trying laplacian with the cv.Laplacian function.
    :param input_file:
    :return:
    """
    # convert to grayscale
    img = cv.imread(input_file, 0)
    edges = cv.Laplacian(img, cv.CV_64F)
    edges = np.absolute(edges)
    edges = np.uint8(edges)

    ret, th2 = cv.threshold(edges, np.min(edges), np.max(edges), cv.THRESH_BINARY + cv.THRESH_OTSU)
    return edges



def borders7(input_file):
    """
    Gaussian filter + Canny + Dilation + Original image
    :param input_file:
    :return:
    """
    img = cv.imread(input_file, 0)
    img = cv.GaussianBlur(img, (5, 5), 0)
    edges = cv.Canny(img, 80, 150)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)
    return edges


def borders8(reduced_img):
    """
    Gaussian filter + Canny + Dilation + Reduced image
    :param input_file:
    :return:
    """
    img = cv.cvtColor(reduced_img, cv.COLOR_BGR2GRAY)
    img = cv.GaussianBlur(img, (5, 5), 0)
    edges = cv.Canny(img, 70, 130)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)
    return edges


def apply_borders(edges, img):
    """
    Binarizes the edges image and then applies it on top of img
    :param edges: output of the edge detection process
    :param img: image on top of which we'll apply the borders
    :return:
    """
    edges = cv.bitwise_not(edges)
    binarized_edges = edges / 255

    # add borders on img by turning black those pixels (zero is black)
    binarized_edges = np.repeat(binarized_edges[:, :, np.newaxis], 3, axis=2)
    return np.multiply(binarized_edges, img)


def cartoonize(input_file):
    """
    Section 3.2.2
    :param input_file:
    :return:
    """
    img = cv.imread(input_file, 1)
    bw_img = cv.imread(input_file, cv.IMREAD_GRAYSCALE)

    # get filename without extension
    filename = os.path.splitext(os.path.basename(input_file))[0]
    ext = os.path.splitext(os.path.basename(input_file))[1]

    # reduce colors
    reduced = reduce_color_convolution(img)
    reduced = reduce_color_proposed(reduced)

    # render borders
    final1 = apply_borders(borders1(input_file), reduced)
    final2 = apply_borders(borders2(reduced), reduced)
    final3 = apply_borders(borders3(input_file), reduced)
    final4 = apply_borders(borders4(input_file), reduced)
    final5 = apply_borders(borders5(input_file), reduced)
    final7 = apply_borders(borders7(input_file), reduced)
    final8 = apply_borders(borders8(reduced), reduced)

    cv.imwrite('output/cartoonize/' + filename + '-3-2-2-var1' + ext, final1)
    cv.imwrite('output/cartoonize/' + filename + '-3-2-2-var2' + ext, final2)
    cv.imwrite('output/cartoonize/' + filename + '-3-2-2-var3' + ext, final3)
    cv.imwrite('output/cartoonize/' + filename + '-3-2-2-var4' + ext, final4)
    cv.imwrite('output/cartoonize/' + filename + '-3-2-2-var5' + ext, final5)
    cv.imwrite('output/cartoonize/' + filename + '-3-2-2-var7' + ext, final7)
    cv.imwrite('output/cartoonize/' + filename + '-3-2-2-var8' + ext, final8)


def main():
    """
    We are trying a few different variations:
    variation 1: color reduction + convolution edge detector on reduced image
    variation 2: color reduction + canny edge detector on reduced image
    variation 3: color reduction + canny edge detector on original image colored
    variation 4: color reduction + canny edge detector on original image black and white
    variation 5: color reduction + canny edge detector on original image + morphological transformation close
    variation 6: DISCARDED
    variation 7: gaussian filter before canny for edge detection
    variation 8: Gaussian filter + Canny + Dilation + Reduced image
    :return:
    """
    input_files = glob.glob("input/*")
    for f in input_files:
        cartoonize(f)


if __name__ == "__main__":
    main()
