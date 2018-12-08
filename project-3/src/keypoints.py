import cv2
import numpy as np


def good_features(img):
    """
    Experiment to detect keypoints based on goodFeaturesToTrack
    Reference https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_shi_tomasi/py_shi_tomasi.html
    """
    img = np.copy(img)
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
    corners = np.int0(corners)
    print('Good Features - number of keypoints:', corners.shape[0])

    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 2, [0, 255, 0], -1)

    cv2.imwrite('output/keypoints/good_features.jpg', img)


def harris(img):
    """
    Experiment to detect keypoints based on the Harris Corner Detector
    Reference: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
    """
    HARRIS_HIGH_THRESHOLD = 0.1
    HARRIS_LOW_THRESHOLD = 0.01

    img = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    kp = dst > HARRIS_HIGH_THRESHOLD * dst.max()
    print('Harris Corner detector (high threshold) - number of keypoints:', np.argwhere(kp == True).shape[0])
    kp = dst > HARRIS_LOW_THRESHOLD * dst.max()
    print('Harris Corner detector (low threshold) - number of keypoints:', np.argwhere(kp == True).shape[0])

    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)

    # High Threshold
    kp = dst > HARRIS_HIGH_THRESHOLD * dst.max()
    img[kp] = [0, 255, 0]
    cv2.imwrite('output/keypoints/harris-high-threshold.jpg', img)

    # Low threshold
    kp = dst > HARRIS_LOW_THRESHOLD * dst.max()
    img[kp] = [0, 255, 0]
    cv2.imwrite('output/keypoints/harris-low-threshold.jpg', img)



def orb(img):
    img = np.copy(img)
    GOOD_MATCH_PERCENT = 0.3
    MAX_FEATURES = 750
    orb = cv2.ORB_create(MAX_FEATURES)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints1, descriptors1 = orb.detectAndCompute(gray, None)

    points1 = np.zeros((len(keypoints1), 2), dtype=np.float32)

    for i, match in enumerate(keypoints1):
        points1[i, :] = match.pt

    points1 = points1.round().astype(int)

    img[points1] = [0, 255, 0]
    cv2.imwrite('output/keypoints/orb.jpg', img)

def sift(img):
    img = np.copy(img)
    sift = cv2.xfeatures2d.SIFT_create(sigma=0.5)
    kp1, des1 = sift.detectAndCompute(img, None)
    kp1 = np.array([i.pt for i in kp1])
    kp1 = kp1.round().astype(int)

    img[kp1] = [0, 255, 0]

    cv2.imwrite('output/keypoints/sift.jpg', img)


def main():

    # Take first frame and find corners in it
    img = cv2.imread('input/hotel.seq/hotel.seq1.png')

    good_features(img)

    harris(img)

    # orb(img)
    #
    # sift(img)


if __name__ == "__main__":
    main()
