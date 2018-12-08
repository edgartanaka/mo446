""""
State University of Campinas

MO446 - Computer Vision (Prof. Adin)
Project 1 - Cartoonization

Authors:
Darley Barreto
Edgar Tanaka
Tiago Barros

Scope of this file:
Test convolutions implemented manually and using OpenCV
"""
import numpy as np
import cv2
import time
from os.path import dirname, abspath

d = dirname(dirname(abspath(__file__)))


def conv_2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel = np.flipud(np.fliplr(kernel))

    kernel_height: int = kernel.shape[0]
    kernel_width: int = kernel.shape[1]

    img_height: int = img.shape[0]
    img_width: int = img.shape[1]
    if kernel_width % 2 == 1:
        kerneloffset: int = (kernel_width - 1) // 2
    else:
        raise Exception("Kernel is not a square matrix of odd size")

    img_padded: np.ndarray = np.zeros((img.shape[0] + 2 * kerneloffset, img.shape[1] + 2 * kerneloffset, img.shape[-1]))
    img_padded[kerneloffset:img.shape[0] + kerneloffset, kerneloffset:img.shape[1] + kerneloffset] = img
    output: np.ndarray = np.zeros_like(img)

    for irow in range(img_height):
        for icolumn in range(img_width):
            for c in range(img.shape[-1]):
                pixel_value: int = (
                            kernel * img_padded[irow:irow + kernel_height, icolumn:icolumn + kernel_width, c]).sum()
                output[irow, icolumn, c] = 0 if pixel_value < 0 else pixel_value if pixel_value < 255 else 255

    return output


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__)
            kw['log_time'][name] = (te - ts) * 1000  # ms
        return result

    return timed


@timeit
def myconv_exp_1(**kwargs):
    # sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    img = cv2.imread(f'{d}/input/dog.jpg', cv2.IMREAD_COLOR)
    output = conv_2d(img, kernel)

    cv2.imwrite(f'{d}/output/convolution/dog-2-1.jpg', output)


@timeit
def myconv_exp_2(**kwargs):
    # emboss
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])

    img = cv2.imread(f'{d}/input/lena.jpg', cv2.IMREAD_COLOR)
    output = conv_2d(img, kernel)

    cv2.imwrite(f'{d}/output/convolution/lena-2-2.jpg', output)


@timeit
def myconv_exp_3(**kwargs):
    # unsharp masking
    kernel = -1 / 256 * np.array([[1, 4, 6, 4, 1],
                                  [4, 16, 24, 16, 4],
                                  [6, 24, -476, 24, 6],
                                  [4, 16, 24, 16, 4],
                                  [1, 4, 6, 4, 1]])

    img = cv2.imread(f'{d}/input/peppers.png', cv2.IMREAD_COLOR)
    output = conv_2d(img, kernel)

    cv2.imwrite(f'{d}/output/convolution/peppers-2-3.png', output)


@timeit
def myconv_exp_4(**kwargs):
    # gaussian blur

    kernel = np.array([[0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036],
                       [0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363],
                       [0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446],
                       [0.002291, 0.023226, 0.092651, 0.146768, 0.092651, 0.023226, 0.002291],
                       [0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446],
                       [0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363],
                       [0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036]])

    img = cv2.imread(f'{d}/input/lake.jpg', cv2.IMREAD_COLOR)
    output = conv_2d(img, kernel)

    cv2.imwrite(f'{d}/output/convolution/lake-2-4.jpg', output)


@timeit
def cv_exp_1(**kwargs):
    # sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    img = cv2.imread(f'{d}/input/dog.jpg', cv2.IMREAD_COLOR)
    output = cv2.filter2D(img, -1, kernel)

    cv2.imwrite(f'{d}/output/convolution/dog-2-1-cv.jpg', output)


@timeit
def cv_exp_2(**kwargs):
    # emboss
    kernel = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]])

    img = cv2.imread(f'{d}/input/lena.jpg', cv2.IMREAD_COLOR)
    output = cv2.filter2D(img, -1, kernel)

    cv2.imwrite(f'{d}/output/convolution/lena-2-2-cv.jpg', output)


@timeit
def cv_exp_3(**kwargs):
    # unsharp masking
    kernel = -1 / 256 * np.array([[1, 4, 6, 4, 1],
                                  [4, 16, 24, 16, 4],
                                  [6, 24, -476, 24, 6],
                                  [4, 16, 24, 16, 4],
                                  [1, 4, 6, 4, 1]])

    img = cv2.imread(f'{d}/input/peppers.png', cv2.IMREAD_COLOR)
    output = cv2.filter2D(img, -1, kernel)

    cv2.imwrite(f'{d}/output/convolution/peppers-2-3-cv.png', output)


@timeit
def cv_exp_4(**kwargs):
    # gaussian blur

    kernel = np.array([[0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036],
                       [0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363],
                       [0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446],
                       [0.002291, 0.023226, 0.092651, 0.146768, 0.092651, 0.023226, 0.002291],
                       [0.001446, 0.014662, 0.058488, 0.092651, 0.058488, 0.014662, 0.001446],
                       [0.000363, 0.003676, 0.014662, 0.023226, 0.014662, 0.003676, 0.000363],
                       [0.000036, 0.000363, 0.001446, 0.002291, 0.001446, 0.000363, 0.000036]])

    img = cv2.imread(f'{d}/input/lake.jpg', cv2.IMREAD_COLOR)
    output = cv2.filter2D(img, -1, kernel)

    cv2.imwrite(f'{d}/output/convolution/lake-2-4-cv.jpg', output)


def main():
    logtime_data = {}

    myconv_exp_1(log_time=logtime_data)
    myconv_exp_2(log_time=logtime_data)
    myconv_exp_3(log_time=logtime_data)
    myconv_exp_4(log_time=logtime_data)

    cv_exp_1(log_time=logtime_data)
    cv_exp_2(log_time=logtime_data)
    cv_exp_3(log_time=logtime_data)
    cv_exp_4(log_time=logtime_data)

    with open("./output/convolution/2-text_output.txt", "w") as f:
        f.write("Report on running time of section 2\n")
        f.write(35 * "=" + "\n\n")
        for idx, i in enumerate(logtime_data):
            expt = i.split("exp_")[-1]

            if "myconv" in i:
                f.write("Our implementation of convolution on experiment {}: {:.3f} ms".format(expt, logtime_data[i]))
            else:
                f.write(
                    "Opencv implementation of convolution on experiment {}: {:.3f} ms".format(expt, logtime_data[i]))

            if idx < len(logtime_data) - 1:
                f.write("\n\n")


if __name__ == "__main__":
    main()
