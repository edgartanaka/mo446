""""
State University of Campinas

MO446 - Computer Vision (Prof. Adin)
Project 1 - Cartoonization

Group 4:
Darley Barreto
Edgar Tanaka
Tiago Barros

Scope of this file:
Reduce color space of images
"""
import matplotlib

matplotlib.use('Agg')
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def clusterize(img: np.ndarray, k: int) -> np.ndarray:
    """
    Function to perform clusterization in an image using k-means
    """

    # Reshaping the input image
    img_reshape = img.reshape((-1, 3))
    img_reshape = np.float32(img_reshape)

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Clusterization process
    ret, label, center = cv2.kmeans(img_reshape, k, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)

    # Generates output image using the centers of clusters
    center = np.uint8(center)
    output = center[label.flatten()]
    output = output.reshape((img.shape))

    return output


def median_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """
    Function to apply a median blur filter in an image
    """
    output = cv2.medianBlur(img, ksize)
    return output


def gaussian(img, ksize):
    return cv2.GaussianBlur(img, (ksize, ksize), -1)


def box_filter(img, ksize):
    return cv2.boxFilter(img, -1, (ksize, ksize))


def print_rgb_histogram(img, file_path, title):
    '''
    Print color histogram
    '''
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], mask=None, histSize=[256], ranges=[0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    plt.title(title)
    plt.savefig(file_path)
    plt.clf()


def print_hsv_histogram(img, file_path, title):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    plt.imshow(hist, interpolation='nearest')
    plt.title(title)
    plt.savefig(file_path)
    plt.clf()


def color_reduction_convolutions():
    """
    Section 3.1.1
    :return:
    """
    input_files = glob.glob("input/*")
    for f in input_files:
        img = cv2.imread(f, 1)

        img_name = os.path.splitext(os.path.basename(f))[0]
        ext = os.path.splitext(os.path.basename(f))[1]

        # save histogram for original image
        print_rgb_histogram(img, os.path.join('output', 'color', img_name + '-rgbhist.png'), img_name + ': Original')
        print_hsv_histogram(img, os.path.join('output', 'color', img_name + '-hsvhist.png'), img_name + ': Original')

        # gaussian kernels
        kernel_sizes = [7, 9, 15, 21]
        for k in kernel_sizes:
            blurred = gaussian(img, k)
            cv2.imwrite('output/color/' + img_name + '-3-1-1-gaussian' + str(k) + ext, blurred)
            print_rgb_histogram(blurred,
                                os.path.join('output', 'color', img_name + '-3-1-1-gaussian' + str(k) + '-rgbhist.png'),
                                img_name + ': Gaussian blur with kernel size ' + str(k))
            print_hsv_histogram(blurred,
                                os.path.join('output', 'color', img_name + '-3-1-1-gaussian' + str(k) + '-hsvhist.png'),
                                img_name + ': Gaussian blur with kernel size ' + str(k)
                                )

        # box filter kernels
        kernel_sizes = [3, 5, 9, 15]
        for k in kernel_sizes:
            blurred = box_filter(img, k)
            cv2.imwrite('output/color/' + img_name + '-3-1-1-box' + str(k) + ext, blurred)
            print_rgb_histogram(blurred,
                                os.path.join('output', 'color', img_name + '-3-1-1-box' + str(k) + '-rgbhist.png'),
                                img_name + ': Box filter with kernel size ' + str(k))
            print_hsv_histogram(blurred,
                                os.path.join('output', 'color', img_name + '-3-1-1-box' + str(k) + '-hsvhist.png'),
                                img_name + ': Box filter with kernel size ' + str(k)
                                )


def color_reduction_clustering():
    """
    Section 3.1.2
    :return: None
    """
    img = cv2.imread(os.path.join('input', 'dog.jpg'), cv2.IMREAD_COLOR)
    img = median_blur(img, 7)
    img = clusterize(img, 9)
    cv2.imwrite(os.path.join('output', 'color', 'dog-3-1-2.jpg'), img)


def main():
    color_reduction_convolutions()
    color_reduction_clustering()


if __name__ == "__main__":
    main()
