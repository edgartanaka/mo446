""""
University of Campinas

MO446 - Computer Vision (Prof. Adin)
Project 2 - Augmented Reality

Authors:
Darley Barreto
Edgar Tanaka
Tiago Barros

Scope of this file:
Implementation of Scale-Invariant Feature Transform (SIFT)

Based on C++ implementation by Utkarsh Sinha:
https://github.com/aishack/sift
"""

import numpy as np
import cv2

class SIFT:
    SIGMA1 = 0.7
    SIGMA2 = 1.2
    DESC_NUM_BINS = 8
    NUM_HIST_BINS = 36
    MAX_KERNEL_SIZE = 20
    PIXEL_THRESHOLD = 0.03
    CURVATURE_THRESHOLD = 12.0
    FEATURE_WINDOW_SIZE = 16
    FEATURE_VECTOR_SIZE = 128
    FEATURE_VECTOR_THRESHOLD = 0.2

    def __init__(self, image: np.ndarray, octaves: int, scales: int):
        """Constructor for the SIFT class"""

        self.image   = image.copy()
        self.octaves = octaves
        self.scales  = scales

        # Creates a 2D array to store the images of the pyramid
        self.arrayImage = np.zeros((self.octaves, self.scales + 3), np.ndarray)

        # Creates a 2D array to store the sigma values used in Gaussian blur
        self.arraySigma = np.zeros((self.octaves, self.scales + 3), np.float64)

        # Creates a 2D array to store the differences of Gaussians
        self.arrayDoG = np.zeros((self.octaves, self.scales + 2), np.ndarray)

        # Creates a 2D array to tell if a particular point is an extrema or not
        self.arrayExtrema = np.zeros((self.octaves, self.scales), np.ndarray)

        # Keeps track of the number of keypoints detected
        self.keypointsNumber  = 0
        self.keypointsRemoved = 0

        # Stores information about the keypoints
        self.keypoints = np.array((), Keypoint)

        # Stores information about the keypoints' descriptors
        self.descriptors = np.array((), Descriptor)

    def saveImage(self, filename: str, image: np.ndarray):
        """Saves a floating-point image as a visible image"""

        cv2.imwrite(filename, cv2.convertScaleAbs(image, None, 255.0))

    def detect(self):
        """Executes all the necessary steps to detect keypoints"""

        # Checks if the keypoint detection was already performed
        if not isinstance(self.arrayImage[0, 0], int):
            return self.keypoints

        self.createScaleSpace()
        self.detectMinimaAndMaxima()
        self.getOrientations()

        return self.keypoints

    def compute(self):
        """Computes the feature descriptors"""

        # Checks if the feature description was already performed
        if len(self.descriptors) != 0:
            return self.keypoints, self.featureVectors

        # Checks if the keypoint detection was made
        if isinstance(self.arrayImage[0, 0], int):
            print("Please, execute \"detect\" method first.")
            print("Or execute only \"detectAndCompute\" method.")
            return None, None

        self.getDescriptors()

        return self.keypoints, self.featureVectors

    def detectAndCompute(self):
        """Executes all the necessary steps to detect keypoints and compute
           feature descriptors"""

        # Checks if the feature description was already performed
        if len(self.descriptors) != 0:
            return self.keypoints, self.featureVectors

        self.createScaleSpace()
        self.detectMinimaAndMaxima()
        self.getOrientations()
        self.getDescriptors()

        return self.keypoints, self.featureVectors

    def createScaleSpace(self):
        """Creates the scale space to find the keypoints"""

        # Transforms the image in a gray image, if it is not already
        if len(self.image.shape) == 3 and self.image.shape[2] >= 3:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.image.copy()

        # Gets an image with floating point values between 0 and 1
        gray_float = gray / 255.0

        # Applies a Gaussian blur to the image
        gray_float = cv2.GaussianBlur(gray_float, (0, 0), self.SIGMA1)

        # Doubles the size of the image
        base = cv2.pyrUp(gray_float)

        base = cv2.GaussianBlur(base, (0, 0), self.SIGMA2)

        # Stores the base image in the pyramid
        self.arrayImage[0, 0] = base

        # Initial base value for sigma
        initialSigma = np.sqrt(2)

        # Stores the first sigma value in the array
        self.arraySigma[0, 0] = 0.5 * initialSigma

        # Generates the other images of the pyramid
        for o in range(self.octaves):

            # Resets the sigma value for each octave
            sigma = initialSigma

            for s in range(1, self.scales + 3):

                # Computes the sigma value to blur the current image
                sigma_f = sigma * np.sqrt(np.power(2, 2/self.scales) - 1)
                sigma   = sigma * np.power(2, 1/self.scales)

                # Keeps track of the sigma values
                self.arraySigma[o, s] = sigma * 0.5 * np.power(2, o)

                # Applies a Gaussian blur to the image
                self.arrayImage[o, s] = cv2.GaussianBlur(self.arrayImage[o, s - 1],
                                                         (0, 0),
                                                         sigma_f)

                # Computes the difference of Gaussians
                self.arrayDoG[o, s - 1] = cv2.subtract(self.arrayImage[o, s - 1],
                                                       self.arrayImage[o, s])

                #self.saveImage('g_octave_{0}_scale_{1}.jpg'.format(o, s),
                #               self.arrayImage[o, s])
                #self.saveImage('dog_octave_{0}_scale_{1}.jpg'.format(o, s-1),
                #               self.arrayDoG[o, s-1])

            # If it is not the last octave, scales the image down
            if o < self.octaves - 1:
                self.arrayImage[o + 1, 0] = cv2.pyrDown(self.arrayImage[o, 0])
                self.arraySigma[o + 1, 0] = self.arraySigma[o, self.scales]

                #self.saveImage('g_octave_{0}_scale_0.jpg'.format(o+1),
                #               self.arrayImage[o+1, 0])

    def detectMinimaAndMaxima(self):
        """Detects the extrema values and keypoints"""

        # Detects the minima and maxima in the differences of Gaussians
        for o in range(self.octaves):
            scale = np.power(2, o)
            for s in range(1, self.scales + 1):

                # Set all points to zero to indicate that it is not a keypoint
                self.arrayExtrema[o, s - 1] = np.zeros(self.arrayDoG[o, 0]
                        .shape, np.uint8)

                upper  = self.arrayDoG[o, s + 1]
                middle = self.arrayDoG[o, s]
                lower  = self.arrayDoG[o, s - 1]

                height, width = middle.shape

                for x in range(1, width - 1):
                    for y in range(1, height - 1):
                        detected = False

                        currentPixel = middle[y, x]

                        if currentPixel > middle[y - 1, x]     and \
                           currentPixel > middle[y + 1, x]     and \
                           currentPixel > middle[y, x - 1]     and \
                           currentPixel > middle[y, x + 1]     and \
                           currentPixel > middle[y - 1, x - 1] and \
                           currentPixel > middle[y - 1, x + 1] and \
                           currentPixel > middle[y + 1, x + 1] and \
                           currentPixel > middle[y + 1, x - 1] and \
                           currentPixel >  upper[y, x]         and \
                           currentPixel >  upper[y - 1, x]     and \
                           currentPixel >  upper[y + 1, x]     and \
                           currentPixel >  upper[y, x - 1]     and \
                           currentPixel >  upper[y, x + 1]     and \
                           currentPixel >  upper[y - 1, x - 1] and \
                           currentPixel >  upper[y - 1, x + 1] and \
                           currentPixel >  upper[y + 1, x + 1] and \
                           currentPixel >  upper[y + 1, x - 1] and \
                           currentPixel >  lower[y, x]         and \
                           currentPixel >  lower[y - 1, x]     and \
                           currentPixel >  lower[y + 1, x]     and \
                           currentPixel >  lower[y, x - 1]     and \
                           currentPixel >  lower[y, x + 1]     and \
                           currentPixel >  lower[y - 1, x - 1] and \
                           currentPixel >  lower[y - 1, x + 1] and \
                           currentPixel >  lower[y + 1, x + 1] and \
                           currentPixel >  lower[y + 1, x - 1]:

                            self.arrayExtrema[o, s - 1][y, x] = 255
                            self.keypointsNumber += 1
                            detected = True

                        elif currentPixel < middle[y - 1, x]     and \
                             currentPixel < middle[y + 1, x]     and \
                             currentPixel < middle[y, x - 1]     and \
                             currentPixel < middle[y, x + 1]     and \
                             currentPixel < middle[y - 1, x - 1] and \
                             currentPixel < middle[y - 1, x + 1] and \
                             currentPixel < middle[y + 1, x + 1] and \
                             currentPixel < middle[y + 1, x - 1] and \
                             currentPixel <  upper[y, x]         and \
                             currentPixel <  upper[y - 1, x]     and \
                             currentPixel <  upper[y + 1, x]     and \
                             currentPixel <  upper[y, x - 1]     and \
                             currentPixel <  upper[y, x + 1]     and \
                             currentPixel <  upper[y - 1, x - 1] and \
                             currentPixel <  upper[y - 1, x + 1] and \
                             currentPixel <  upper[y + 1, x + 1] and \
                             currentPixel <  upper[y + 1, x - 1] and \
                             currentPixel <  lower[y, x]         and \
                             currentPixel <  lower[y - 1, x]     and \
                             currentPixel <  lower[y + 1, x]     and \
                             currentPixel <  lower[y, x - 1]     and \
                             currentPixel <  lower[y, x + 1]     and \
                             currentPixel <  lower[y - 1, x - 1] and \
                             currentPixel <  lower[y - 1, x + 1] and \
                             currentPixel <  lower[y + 1, x + 1] and \
                             currentPixel <  lower[y + 1, x - 1]:
                            self.arrayExtrema[o, s - 1][y, x] = 255
                            self.keypointsNumber += 1
                            detected = True

                        if detected:
                            self.removeBadKeypoints(middle, x, y,
                                    self.arrayExtrema[o, s - 1])

                #self.saveImage('extrema_octave_{0}_scale_{1}.jpg'.format(o,s-1),
                #               self.arrayExtrema[o, s-1])

        print("Keypoints found: {0}".format(self.keypointsNumber))
        print("Keypoints rejected: {0}".format(self.keypointsRemoved))

    def removeBadKeypoints(self, middle: np.ndarray, x: int, y: int,
            extrema: np.ndarray):
        """Removes the extrema values that are not good keypoints"""

        # Checks if the pixel value is less than the threshold
        if np.fabs(middle[y, x]) < self.PIXEL_THRESHOLD:
            extrema[y, x] = 0
            self.keypointsNumber -= 1
            self.keypointsRemoved += 1
            return

        # Computes the curvature threshold using the constant value provided
        curvatureThreshold = ((self.CURVATURE_THRESHOLD + 1) ** 2) \
                           /   self.CURVATURE_THRESHOLD

        # Checks if the pixel is on an edge
        dxx = middle[y - 1, x] + middle[y + 1, x] - 2 * middle[y, x]
        dyy = middle[y, x - 1] + middle[y, x + 1] - 2 * middle[y, x]
        dxy = (middle[y - 1, x - 1] + middle[y + 1, x + 1] \
            -  middle[y + 1, x - 1] - middle[y - 1, x + 1]) / 4.0

        traceH = dxx + dyy
        detH   = dxx * dyy - dxy ** 2

        curvatureRatio = (traceH ** 2) / detH

        if detH < 0 or curvatureRatio > curvatureThreshold:
            extrema[y, x] = 0
            self.keypointsNumber -= 1
            self.keypointsRemoved += 1

    def getOrientations(self):
        """Assign orientations for keypoints"""

        # Creates 2D arrays to store the magnitudes and orientations of
        # gradients of the images
        magnitude   = np.zeros((self.octaves, self.scales), np.ndarray)
        orientation = np.zeros((self.octaves, self.scales), np.ndarray)

        for o in range(self.octaves):
            for s in range(1, self.scales + 1):
                magnitude[o, s - 1]   = np.zeros(self.arrayImage[o, s].shape,
                                                 np.float32)
                orientation[o, s - 1] = np.zeros(self.arrayImage[o, s].shape,
                                                 np.float32)

                height, width = self.arrayImage[o, s].shape

                for x in range(1, width - 1):
                    for y in range(1, height - 1):

                        # Computes the gradient
                        dx = self.arrayImage[o, s][y, x + 1] \
                           - self.arrayImage[o, s][y, x - 1]
                        dy = self.arrayImage[o, s][y + 1, x] \
                           - self.arrayImage[o, s][y - 1, x]

                        magnitude[o, s - 1][y, x] = np.sqrt(dx ** 2 + dy ** 2)

                        if dx == 0:
                            if dy == 0:
                                ratio = 0
                            elif dy > 0:
                                ratio = np.inf
                            else:
                                ratio = -np.inf
                        else:
                            ratio = dy / dx

                        orientation[o, s - 1][y, x] = np.arctan(ratio)

                #self.saveImage('mag_octave_{0}_scale{1}.jpg'.format(o, s-1),
                #               magnitude[o, s-1])
                #self.saveImage('ori_octave_{0}_scale{1}.jpg'.format(o, s-1),
                #               orientation[o, s-1])

        # Creates the histogram of orientation
        histogram = np.zeros(self.NUM_HIST_BINS, np.float64)

        for o in range(self.octaves):
            scale = np.power(2, o)
            height, width = self.arrayImage[o, 0].shape
            for s in range(1, self.scales + 1):
                sigma = self.arraySigma[o, s]

                imgWeight = cv2.GaussianBlur(magnitude[o, s-1],
                                             (0, 0),
                                             1.5 * sigma)

                hfsz = int(self.getKernelSize(1.5 * sigma) / 2)

                imgMask = np.zeros(self.arrayImage[o, 0].shape, np.uint8)

                for x in range(width):
                    for y in range(height):
                        if self.arrayExtrema[o, s - 1][y, x] != 0:
                            for kk in range(-hfsz, hfsz + 1):
                                for tt in range(-hfsz, hfsz + 1):
                                    if x + kk < 0 or x + kk >= width or \
                                       y + tt < 0 or y + tt >= height:
                                        continue

                                    orient = orientation[o, s - 1][y + tt, x + kk]

                                    if orient <= -np.pi or orient > np.pi:
                                        print("Bad orientation: {0}"
                                                .format(orient))

                                    orient += np.pi

                                    # Convert the orientation to degrees
                                    orientDegrees = orient * 180 / np.pi
                                    histogram[int(orientDegrees / (360 /
                                        self.NUM_HIST_BINS))] += imgWeight[y +
                                                tt, x + kk]
                                    imgMask[y + tt, x + kk] = 255

                            # Checks the maximum of the histogram
                            maxValue = histogram.max()

                            # Lists of magnitudes and orientations
                            # at the current extrema
                            magnitudes   = np.array((), np.float64)
                            orientations = np.array((), np.float64)

                            for k in range(self.NUM_HIST_BINS):
                                if histogram[k] > 0.8 * maxValue:
                                    x1 = k - 1
                                    x2 = k
                                    x3 = k + 1
                                    y2 = histogram[k]

                                    if k == 0:
                                        y1 = histogram[self.NUM_HIST_BINS - 1]
                                        y3 = histogram[1]
                                    elif k == self.NUM_HIST_BINS - 1:
                                        y1 = histogram[self.NUM_HIST_BINS - 1]
                                        y3 = histogram[0]
                                    else:
                                        y1 = histogram[k - 1]
                                        y3 = histogram[k + 1]

                                    X = np.zeros((3, 3), np.float32)

                                    X[0, 0] = x1 ** 2
                                    X[1, 0] = x1
                                    X[2, 0] = 1

                                    X[0, 1] = x2 ** 2
                                    X[1, 1] = x2
                                    X[2, 1] = 1

                                    X[0, 2] = x3 ** 2
                                    X[1, 2] = x3
                                    X[2, 2] = 1

                                    inverse = cv2.invert(X)

                                    b = np.zeros(3, np.float64)

                                    b[0] = y1 * inverse[1][0, 0] \
                                         + y2 * inverse[1][1, 0]
                                    b[1] = y1 * inverse[1][0, 1] \
                                         + y2 * inverse[1][1, 1]
                                    b[2] = y1 * inverse[1][0, 2] \
                                         + y2 * inverse[1][1, 2]

                                    x0 = -b[1] / (2 * b[0])

                                    if np.abs(x0) > 2 * self.NUM_HIST_BINS:
                                        x0 = x2

                                    while x0 < 0:
                                        x0 += self.NUM_HIST_BINS
                                    while x0 >= self.NUM_HIST_BINS:
                                        x0 -= self.NUM_HIST_BINS

                                    x0_norm = x0 * (2 * np.pi / self.NUM_HIST_BINS)

                                    assert x0_norm >= 0 and x0_norm < 2 * np.pi
                                    x0_norm -= np.pi
                                    assert x0_norm >= -np.pi and x0_norm < np.pi

                                    magnitudes = np.append(magnitudes,
                                            histogram[k])
                                    orientations = np.append(orientations,
                                            x0_norm)

                            self.keypoints = np.append(self.keypoints,
                                    Keypoint(x * scale / 2, y * scale / 2,
                                        magnitudes, orientations,
                                        o * self.scales + s - 1))

                #self.saveImage('ori_region_octave_{0}_scale{1}.jpg'.format(o,
                #    s-1), imgMask)

        # Sanity check
        assert len(self.keypoints) == self.keypointsNumber

        # Array to store the feature vectors
        self.featureVectors = np.zeros((self.keypointsNumber,
                                        self.FEATURE_VECTOR_SIZE), np.float32)

    def getDescriptors(self):
        """Computes the feature descriptors"""

        interpolatedMagnitude   = np.zeros((self.octaves, self.scales), np.ndarray)
        interpolatedOrientation = np.zeros((self.octaves, self.scales), np.ndarray)

        for o in range(self.octaves):
            for s in range(1, self.scales + 1):
                height, width = self.arrayImage[o, s].shape

                imgTmp = cv2.pyrUp(self.arrayImage[o, s])

                interpolatedMagnitude[o, s - 1]   = np.zeros((height + 1, width
                    + 1), np.float32)
                interpolatedOrientation[o, s - 1] = np.zeros((height + 1, width
                    + 1), np.float32)

                i = 1.5

                while i < width - 1.5:
                    j = 1.5

                    while j < height - 1.5:
                        dx = ((self.arrayImage[o, s][int(j), int(i + 1.5)]
                           +   self.arrayImage[o, s][int(j), int(i + 0.5)]) / 2
                           -  (self.arrayImage[o, s][int(j), int(i - 1.5)]
                           +   self.arrayImage[o, s][int(j), int(i - 0.5)]) / 2)
                        dy = ((self.arrayImage[o, s][int(j + 1.5), int(i)]
                           +   self.arrayImage[o, s][int(j + 0.5), int(i)]) / 2
                           -  (self.arrayImage[o, s][int(j - 1.5), int(i)]
                           +   self.arrayImage[o, s][int(j - 0.5), int(i)]) / 2)

                        ii = int(i + 1)
                        jj = int(j + 1)

                        assert ii <= width and jj <= height

                        interpolatedMagnitude[o, s - 1][jj, ii] = np.sqrt(dx **
                                2 + dy ** 2)
                        interpolatedOrientation[o, s - 1][jj, ii] = (-np.pi if
                                np.arctan2(dy, dx) == np.pi else
                                np.arctan2(dy, dx))

                        j += 1
                    i += 1

                for ii in range(width + 1):
                    interpolatedMagnitude  [o, s - 1][0, ii] = 0
                    interpolatedMagnitude  [o, s - 1][height, ii] = 0
                    interpolatedOrientation[o, s - 1][0, ii] = 0
                    interpolatedOrientation[o, s - 1][height, ii] = 0

                for jj in range(height + 1):
                    interpolatedMagnitude  [o, s - 1][jj, 0] = 0
                    interpolatedMagnitude  [o, s - 1][jj, width] = 0
                    interpolatedOrientation[o, s - 1][jj, 0] = 0
                    interpolatedOrientation[o, s - 1][jj, width] = 0

                #self.saveImage('intmag_octave_{0}_scale_{1}.jpg'.format(o,
                #    s - 1), interpolatedMagnitude[o, s - 1])
                #self.saveImage('intori_octave_{0}_scale_{1}.jpg'.format(o,
                #    s - 1), interpolatedOrientation[o, s - 1])

        G = self.getInterpolatedGaussianTable(self.FEATURE_WINDOW_SIZE,
                                        0.5 * self.FEATURE_WINDOW_SIZE)

        histogram = np.zeros(self.DESC_NUM_BINS, np.float64)

        for kp in range(self.keypointsNumber):
            scale = self.keypoints[kp].scale
            kpx   = self.keypoints[kp].pt[0]
            kpy   = self.keypoints[kp].pt[1]

            ii = int(2 * kpx / np.power(2, float(scale) / float(self.scales)))
            jj = int(2 * kpy / np.power(2, float(scale) / float(self.scales)))

            height, width = self.arrayImage[int(scale / self.scales), 0].shape

            magnitude   = self.keypoints[kp].magnitude
            orientation = self.keypoints[kp].orientation

            mainMagnitude   = magnitude[np.argmax(magnitude)]
            mainOrientation = orientation[np.argmax(magnitude)]

            hfsz = int(0.5 * self.FEATURE_WINDOW_SIZE)

            weight = np.zeros((self.FEATURE_WINDOW_SIZE,
                               self.FEATURE_WINDOW_SIZE), np.float32)

            # Creates the feature vector
            featureVector = np.zeros(self.FEATURE_VECTOR_SIZE, np.float64)

            for index, value in np.ndenumerate(weight):
                if ii + index[0] + 1 < hfsz or \
                   ii + index[0] + 1 > width + hfsz or \
                   jj + index[1] + 1 < hfsz or \
                   jj + index[1] + 1 > height + hfsz:
                    weight[index[1], index[0]] = 0
                else:
                    weight[index[1], index[0]] = G[index[1], index[0]] * \
                            interpolatedMagnitude[divmod(scale,
                            self.scales)][jj + index[1] + 1 - hfsz,
                            ii + index[0] + 1 - hfsz]

            for i in range(int(self.FEATURE_WINDOW_SIZE / 4)):
                for j in range(int(self.FEATURE_WINDOW_SIZE / 4)):
                    start_i = ii - hfsz + 1 + int(hfsz / 2 * i)
                    start_j = jj - hfsz + 1 + int(hfsz / 2 * j)
                    limit_i = ii + int(hfsz / 2) * (i - 1)
                    limit_j = jj + int(hfsz / 2) * (j - 1)

                    for k in range(start_i, limit_i):
                        for t in range(start_j, limit_j):
                            if k < 0 or k > width or t < 0 or t > height:
                                continue;

                            # Rotation invariance
                            orient = interpolatedOrientation[divmod(scale,
                                self.scales)][t, k]
                            orient -= mainOrientation

                            while orient < 0:
                                orient += 2 * np.pi

                            while orient > 2 * np.pi:
                                orient -= 2 * np.pi

                            # Sanity check
                            if orient < 0 or orient >= 2 * np.pi:
                                print("Bad orientation: {0}".format(orient))
                            assert orient >= 0 and orient < 2 * np.pi

                            orientDegrees = orient * 180 / np.pi

                            assert orientDegrees < 360

                            bin = int(orientDegrees / (360 /
                                self.DESC_NUM_BINS))
                            binFloat = float(orientDegrees / (360 /
                                self.DESC_NUM_BINS))

                            assert bin < self.DESC_NUM_BINS
                            assert (k + hfsz - 1 - ii < self.FEATURE_WINDOW_SIZE
                               and t + hfsz - 1 - jj < self.FEATURE_WINDOW_SIZE)

                            histogram[bin] += (1 - np.fabs(binFloat - (bin +
                                0.5))) * weight[t + hfsz - 1 - jj, k + hfsz - 1
                                - ii]

                    for t in range(self.DESC_NUM_BINS):
                        featureVector[int((i * self.FEATURE_WINDOW_SIZE / 4 + j)
                            * self.DESC_NUM_BINS + t)] = histogram[t]

            # Normalise the feature vector to ensure illumination independence
            self.normaliseVector(featureVector)

            # Threshold the feature vector
            for t in range(self.FEATURE_VECTOR_SIZE):
                if featureVector[t] > self.FEATURE_VECTOR_THRESHOLD:
                    featureVector[t] = self.FEATURE_VECTOR_THRESHOLD

            # Normalise the feature vector again
            self.normaliseVector(featureVector)

            self.descriptors = np.append(self.descriptors, Descriptor(kpx, kpy,
                featureVector))
            self.featureVectors[kp] = featureVector

        # Sanity check
        assert len(self.descriptors) ==  self.keypointsNumber


    def getKernelSize(self, sigma: float, cutoff: float = 0.001) -> int:
        """Gets the size of the kernel for the Gaussian blur"""

        s = 0

        while s < self.MAX_KERNEL_SIZE:
            if np.exp(-(s ** 2) / (2 * (sigma ** 2))) < cutoff:
                break
            s += 1

        return 2 * s - 1

    def getInterpolatedGaussianTable(self, size: int, sigma: float) -> np.ndarray:
        """Gets the bell curve for the image"""

        assert size % 2 == 0

        halfKernelSize = size / 2 - 0.5

        table = np.zeros((size, size), np.float32)

        sog = 0.0

        for index, value in np.ndenumerate(table):
            tmp = self.getGaussian2D(index[0] - halfKernelSize,
                                     index[1] - halfKernelSize,
                                     sigma)

            table[index[1], index[0]] = tmp
            sog += tmp

        for index, value in np.ndenumerate(table):
            table[index[1], index[0]] = 1.0 / sog * table[index[1], index[0]]

        return table

    def getGaussian2D(self, x: float, y: float, sigma: float) -> float:
        """Gets the value of the bell curve at a point"""

        return 1.0 / (2.0 * np.pi * sigma ** 2) \
                * np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))

    def normaliseVector(self, vector: np.ndarray):
        """Normalises a vector"""

        norm = 0.0

        for i in range(vector.size):
            norm += np.power(vector[i], 2)

        norm = np.sqrt(norm)

        for i in range(vector.size):
            vector[i] /= norm

    def drawKeypoints(self, outputImage: str):
        """Draws the detected keypoints over the input image"""

        image = self.image.copy()

        for i in range(self.keypointsNumber):
            kp = self.keypoints[i]

            cv2.line(image,
                     (int(kp.pt[0]), int(kp.pt[1])),
                     (int(kp.pt[0]), int(kp.pt[1])),
                     (0, 0, 255),
                     3)
            cv2.line(image,
                     (int(kp.pt[0]), int(kp.pt[1])),
                     (int(kp.pt[0] + 10 * np.cos(kp.orientation[0])),
                      int(kp.pt[1] + 10 * np.sin(kp.orientation[0]))),
                     (0, 0, 255),
                     1)

        cv2.imwrite(outputImage, image)

    def printAbsSigma(self):
        """Prints the sigma values used for the images"""

        print("printAbsSigma")

        for o in range(self.octaves):
            for s in range(1, self.scales + 4):
                print(self.arraySigma[o, s - 1], end="\t")

            print()


class Keypoint:
    def __init__(self, x: float, y: float, magnitude: np.array, orientation:
            np.array, scale: np.uint8):
        self.pt = (x, y)
        self.magnitude = magnitude
        self.orientation = orientation
        self.scale = scale

    def __str__(self):
        return "X: {0}, Y:{1}, Magnitude:{2}, Orientation:{3}, Scale:{4}". \
               format(self.pt[0], self.pt[1], self.magnitude, self.orientation,
                      self.scale)


class Descriptor:
    def __init__(self, x: float, y: float, featureVector: np.array):
        self.x = x
        self.y = y
        self.featureVector = featureVector

    def __str__(self):
        return "X: {0}, Y:{1}, Feature Vector:{2}".format(self.x, self.y,
                                                          self.featureVector)
