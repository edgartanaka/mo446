""""
State University of Campinas

MO446 - Computer Vision (Prof. Adin)
Project 1 - Cartoonization

Authors:
Darley Barreto
Edgar Tanaka
Tiago Barros

Scope of this file:
Edge detection
"""
import cv2
import numpy as np


def conv_img1():
    img = cv2.imread('input/lake.jpg', 0)

    # laplacian of gaussian centered on zero
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    output = cv2.filter2D(img, -1, kernel)
    cv2.imwrite('output/edge/lake-3-2-1-1.jpg', output)


def conv_img2():
    img = cv2.imread('input/lena.jpg', 0)

    # sobel Y
    kernel = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]])

    output = cv2.filter2D(img, -1, kernel)
    cv2.imwrite('output/edge/lena-3-2-1-2.jpg', output)


def conv_img3():
    img = cv2.imread('input/peppers.png', 0)

    # sobel X
    kernel = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])

    output = cv2.filter2D(img, -1, kernel)
    cv2.imwrite('output/edge/peppers-3-2-1-3.png', output)


def canny_img1():
    img = cv2.imread('input/lake.jpg', 0)

    output = cv2.Canny(img, 100, 100)

    cv2.imwrite('output/edge/lake-3-2-1-1-canny.jpg', output)


def canny_img2():
    img = cv2.imread('input/lena.jpg', 0)

    output = cv2.Canny(img, 50, 300)

    cv2.imwrite('output/edge/lena-3-2-1-2-canny.jpg', output)


def canny_img3():
    img = cv2.imread('input/peppers.png', 0)

    output = cv2.Canny(img, 70, 200)

    cv2.imwrite('output/edge/peppers-3-2-1-3-canny.png', output)


def main():
    conv_img1()
    conv_img2()
    conv_img3()
    canny_img1()
    canny_img2()
    canny_img3()


if __name__ == "__main__":
    main()
