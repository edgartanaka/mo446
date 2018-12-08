import os
import cv2
import numpy as np
import time
from scipy import signal
import logging

"""
Steps:
- find keypoints with Harris corner 
- for each keypoint
    - create mask 15x15
    - for each point in the mask, calculate gradient in x and in y
    - compute u, v by solving linear system
    - save u, v
    - discard bad keypoints ???
"""

logger = logging.getLogger('klt')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('klt.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

# 'application' code
logger.debug('debug message')

def keypoints(img_gray):
    dst = cv2.cornerHarris(img_gray, 2, 3, 0.04)
    kp = dst > 0.1 * dst.max()

    # return the coordinates x, y of the keypoints
    kp = np.argwhere(kp == True)
    kp = kp.reshape(kp.shape[0], 1, 2)

    print('Total keypoints:', kp.shape[0])

    return kp


def calcLK(prev_img, current_img, kp, window_size):
    """
    Lucas Kanade algorithm for feature tracking
    Inspired by https://sandipanweb.wordpress.com/2018/02/25/implementing-lucas-kanade-optical-flow-algorithm-in-python/
    :param prev_img: gray image
    :param current_img: gray image
    :param kp: list of coordinates x,y
    :return:
    """
    w = int(window_size / 2)

    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])

    mode = 'same'
    Ix = signal.convolve2d(prev_img, kernel_x, boundary='symm', mode=mode)
    Iy = signal.convolve2d(prev_img, kernel_y, boundary='symm', mode=mode)
    It = signal.convolve2d(current_img, kernel_t, boundary='symm', mode=mode) - signal.convolve2d(prev_img, kernel_t, boundary='symm', mode=mode)

    # calculate matrices multiplied
    IxIx = Ix * Ix
    IyIy = Iy * Iy
    IxIy = Ix * Iy
    IxIt = Ix * It
    IyIt = Iy * It

    x_len, y_len = IxIy.shape
    num_keypoints = kp.shape[0]

    # array with all the predicted coordinates of the keypoints after displacement
    predicted = np.empty([num_keypoints, 1, 2])

    # array containing 1 and 0 for each entry in the predicted array
    # 0 are to be discarded, 1 are good points
    status = np.empty([num_keypoints, 1])

    # for each keypoint, calculate displacement [u,v] and add it to the predicted array
    for idx, p in enumerate(kp):
        p_float = p[0]
        p_int = p[0].round().astype(int)
        j, i = p_int[0], p_int[1]

        # if too close to borders, discard it
        if i - w < 0 or j - w < 0 or i + w + 1 >= x_len or j + w + 1 >= y_len:
            predicted[idx] = [0, 0]
            status[idx] = 0
            continue

        # building a and b in linear equation ax = b
        sum_xx = IxIx[i - w:i + w + 1, j - w:j + w + 1].flatten().sum()
        sum_yy = IyIy[i - w:i + w + 1, j - w:j + w + 1].flatten().sum()
        sum_xy = IxIy[i - w:i + w + 1, j - w:j + w + 1].flatten().sum()
        sum_xt = IxIt[i - w:i + w + 1, j - w:j + w + 1].flatten().sum()
        sum_yt = IyIt[i - w:i + w + 1, j - w:j + w + 1].flatten().sum()
        a = np.array([[sum_xx, sum_xy], [sum_xy, sum_yy]])
        b = -np.array([[sum_xt], [sum_yt]])

        # solving for u and v
        # u_v = np.linalg.pinv(a.T.dot(a)).dot(a.T).dot(b)
        u_v = np.linalg.solve(a, b)
        u, v = u_v[0], u_v[1]

        predicted_x = p_float[0] - u
        predicted_y = p_float[1] - v

        # if good point, add it to good
        # otherwise, discard it
        predicted[idx][0] = [predicted_x, predicted_y]

        if 0 <= predicted_y < y_len and 0 <= predicted_x < x_len:
            # good point
            status[idx] = 1
        else:
            # bad point
            status[idx] = 0

    # convert from float64 to float32
    predicted = predicted.astype(np.float32)
    status = status.astype(np.float32)

    return predicted, status, None


def exp1(klt):
    """
    Experiment with slow.flv video of cars in a road
    :return:
    """
    print('KLT Experiment:', klt)

    cap = cv2.VideoCapture('input/slow.flv')

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # list of coordinates
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while True:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        if klt == 'opencv':
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        else:
            p1, st, err = calcLK(old_gray, frame_gray, p0)

        if p1 is None:
            break

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)

        cv2.imshow('frame', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    if klt == 'opencv':
        cv2.imwrite('output/klt/klt-opencv-slow.jpg', img)
    else:
        cv2.imwrite('output/klt/klt-own-slow.jpg', img)

    cv2.destroyAllWindows()
    cap.release()


def exp2(klt, window_size=15):
    logger.debug('Running KLT Experiment 2:' +  klt)

    image_prefix = 'hotel.seq'
    input_frames_dir = 'input/' + image_prefix

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # read first frame
    frame_idx = 1
    filename = os.path.join(input_frames_dir, image_prefix + '1.png')
    old_frame = cv2.imread(filename)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # select keypoints
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    while True:
        frame_idx += 1
        filename = os.path.join(input_frames_dir, image_prefix + str(frame_idx) + '.png')

        frame = cv2.imread(filename)
        if frame is None:
            # already read last frame
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        if klt == 'opencv':
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # logger.debug('OPENCV:' + str(p1))
        else:
            p1, st, err = calcLK(old_gray, frame_gray, p0, window_size)
            # logger.debug('OWN:' + str(p1))

        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

        img = cv2.add(frame, mask)
        if klt == 'opencv':
            cv2.imwrite('output/klt/klt-opencv-hotel.jpg', img)
        else:
            cv2.imwrite('output/klt/klt-own-w' + str(window_size) + '-hotel.jpg', img)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cv2.destroyAllWindows()


def main():
    # exp1('opencv')
    # exp1('myklt')



    logger.debug('------------------------------ OWN KLT -----------------------------------')
    exp2('own', window_size=5)
    exp2('own', window_size=15)
    exp2('own', window_size=30)

    logger.debug('------------------------------ OPENCV KLT -----------------------------------')
    exp2('opencv')


if __name__ == "__main__":
    main()
