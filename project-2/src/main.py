import cv2
import numpy as np
import os
from affine import RANSAC, get_affine
from feature_matching import matching, explore_match

GOOD_MATCH_PERCENT = 0.3
MAX_FEATURES = 750

# Global variables
mask, target = None, None
frame_index = 0


def get_matches_opencv(keypoints1, descriptors1, keypoints2, descriptors2):
    # print('len of descriptors1', len(descriptors1))
    # print('len of descriptors2', len(descriptors2))

    # SIFT
    # bf = cv2.BFMatcher()
    # matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    # good = []
    # points = []
    # for m, n in matches:
    #     if m.distance < 0.75 * n.distance:
    #         points.append([m])
    #         good.append(m)
    # Extract location of good matches
    # points1 = np.zeros((len(good), 2), dtype=np.float32)
    # points2 = np.zeros((len(good), 2), dtype=np.float32)
    #
    # for i, match in enumerate(good):
    #     points1[i, :] = keypoints1[match.queryIdx].pt
    #     points2[i, :] = keypoints2[match.trainIdx].pt
    # return points1, points2


    # ORB
    # TODO: need to replace this for match implemented by Darley
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    img1 = cv2.imread('input/frames/rotate/frame0.jpg', 0)  # queryImage
    img2 = cv2.imread('input/frames/rotate/frame1.jpg', 0)  # trainImage
    # img3 = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, None, flags=2)
    # cv2.imwrite('output/frames/rotate/aaa.jpg', img3)

    # exit(1)
    return points1, points2



def show(img):
    cv2.imshow('dst', img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


def find_four_corners(filename):
    imgray = cv2.imread(filename, 0)

    imgray = cv2.GaussianBlur(imgray, None, 3)
    # show(imgray)

    canny = cv2.Canny(imgray, 50, 120)
    # show(canny)

    # img = cv2.imread(filename)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # h, s, v = cv2.split(hsv)
    # th, thresh = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)

    im2, contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ## sort and choose the largest contour
    cnts = sorted(contours, key=cv2.contourArea)
    cnt = cnts[-1]

    ## approx the contour, so the get the corner points
    arclen = cv2.arcLength(cnt, True)
    img = cv2.imread(filename)
    # cv2.drawContours(canvas, [cnt], -1, (255, 0, 0), 1, cv2.LINE_AA)
    # cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 1, cv2.LINE_AA)
    # show(img)

    # find rotated rectangle
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    # show(img)

    # img = cv2.imread(filename)
    # x, y, w, h = cv2.boundingRect(cnt)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # show(img)

    # cv2.drawContours(imgray, contours, 0, 255, 3)
    # cv2.drawContours(imgray, contours, -1, 255, 3)
    # show(imgray)

    return box


def build_mask_target(corners, first_frame):
    first_frame_img = cv2.imread(first_frame, 0)
    frame_h, frame_w  = first_frame_img.shape
    # show(first_frame_img)

    corners = corners[corners[:, 0].argsort()]
    left = corners[0:2, :]
    left[:, 1].argsort()
    right = corners[2:4, :]
    right[:, 1].argsort()
    lower_left, upper_left, lower_right, upper_right = left[0], left[1], right[0], right[1]

    # load target image
    target_img = cv2.imread("input/target.jpg", 1)  # this is the smaller image we'll paste

    target_h, target_w, _ = target_img.shape

    original = np.array([
        [0, 0],
        [target_w, 0],
        [0, target_h],
        [target_w, target_h]
    ])
    transformed = np.array([
        lower_left,
        lower_right,
        upper_left,
        upper_right
    ])

    _, affine, _ = get_affine(original, transformed)
    affine = affine.reshape(2, 3)

    # build binary mask containing 1s in the location of the target and then warp
    mask = np.zeros((frame_h, frame_w), np.uint8)
    mask[0:target_h, 0:target_w] = 1
    cv2.imwrite('output/mask1.jpg', mask * 255)
    mask = cv2.warpAffine(mask, affine, (mask.shape[1], mask.shape[0]))
    cv2.imwrite('output/mask2.jpg', mask*255)

    # build target with all zeroes but with the target image in its right location and then warp
    target = np.zeros((frame_h, frame_w, 3), np.uint8)
    # print('target_w:', target_w)
    # print('target_h:', target_h)
    # print('target_img shape:', target_img.shape)
    target[0:target_h, 0:target_w] = target_img
    cv2.imwrite('output/target1.jpg', target)
    target = cv2.warpAffine(target, affine, (target.shape[1], target.shape[0]))
    cv2.imwrite('output/target2.jpg', target)

    return mask, target


def process(im1, im2):
    """
    Runs the entire pipeline for a pair of frames:

    :return:
    """
    global mask, target

    # Detect keypoints and Describe them
    sift = cv2.xfeatures2d.SIFT_create(sigma=0.5)
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    kp1 = np.array([i.pt for i in kp1])
    kp2 = np.array([i.pt for i in kp2])

    # print('kp1:', len(kp1))
    # print('kp2:', len(kp2))
    # print('des1:', len(des1))
    # print('des2:', len(des2))

    # Find matches
    points1, points2 = matching(kp1, kp2, des1, des2)
    points1, points2 = RANSAC(points1, points2, 3, 100, 9000, 2)

    # Compute affine transformation matrix
    cols, rows, _ = im1.shape
    _, affine_transform, _ = get_affine(points1, points2)
    affine_transform = affine_transform.reshape(2, 3)
    mask = cv2.warpAffine(mask, affine_transform, (mask.shape[1], mask.shape[0]))
    target = cv2.warpAffine(target, affine_transform, (mask.shape[1], mask.shape[0]))

    # copy only the target image already warped
    final = np.copy(im2)
    np.copyto(final, target, where=mask[:, :, None].astype(bool))

    return final


def main():
    videos = ['rotate', 'scale', 'translate']

    # process frames
    for video in videos:
        global mask, target, corners, frame_index
        mask, target = None, None

        print('\nProcessing frames from ' + video)

        input_frames_dir = 'input/frames/' + video
        output_frames_dir = 'output/frames/' + video

        frames_count = len([f for f in os.listdir(input_frames_dir) if os.path.isfile(os.path.join(input_frames_dir, f))])
        print('Total frames:', frames_count)

        # detect 4 corners with first frame
        first_frame = os.path.join(input_frames_dir, "frame0.jpg")
        corners = find_four_corners(first_frame)

        # process 1st frame
        # assumptions:
        # the 2 with lowest X are the left side
        # the 2 with highest X are the right side
        mask, target = build_mask_target(corners, first_frame)
        final = np.copy(cv2.imread(first_frame, 1))
        np.copyto(final, target, where=mask[:, :, None].astype(bool))
        cv2.imwrite(os.path.join(output_frames_dir, 'frame0.jpg'), final)

        for frame_index in range(1, frames_count-1):
            cur_frame = os.path.join(input_frames_dir, "frame" + str(frame_index-1) + ".jpg")
            next_frame = os.path.join(input_frames_dir, "frame" + str(frame_index) + ".jpg")

            # Read current frame and next
            frame1 = cv2.imread(cur_frame, cv2.IMREAD_COLOR)
            frame2 = cv2.imread(next_frame, cv2.IMREAD_COLOR)
            pasted = process(frame1, frame2)

            # Write processed frame
            cv2.imwrite(os.path.join(output_frames_dir, 'frame' + str(frame_index) + ".jpg"), pasted)

            # cv2.imshow('pasted', pasted)
            # cv2.waitKey(1)

            print('Processed ' + str(frame_index) + ' frames out of ' + str(frames_count))

        cv2.destroyAllWindows()

        print('\nBuilding output video for ' + video)
        frame1 = cv2.imread(os.path.join(output_frames_dir, "frame1.jpg"), cv2.IMREAD_COLOR)
        height, width, layers = frame1.shape

        # Write all output images into one video
        video = cv2.VideoWriter(os.path.join('output', video + '.avi'), cv2.VideoWriter_fourcc(*"MJPG"), 30, (width, height))

        for frame_index in range(frames_count-1):
            video.write(cv2.imread(os.path.join(output_frames_dir, 'frame' + str(frame_index) + ".jpg")))

        cv2.destroyAllWindows()
        video.release()


    print("FINISHED")


if __name__ == "__main__":
    main()
