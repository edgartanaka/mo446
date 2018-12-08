import cv2
import os

# Reduces in half the dimensions of the video frame (to speed up SIFT)
REDUCE = False

def extract_frames(video):
    vidcap = cv2.VideoCapture('input/' + video + '.mp4')
    success, image = vidcap.read()
    count = 0

    print('Success in reading video:', success)

    while success:
        if REDUCE:
            height, width, layers = image.shape
            image = cv2.resize(image, (int(width / 2), int(height / 2)))

        cv2.imwrite(os.path.join('input/frames/' + video, 'frame' + str(count) + '.jpg'),
                    image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Extracted frame', count, 'from video', video)
        count += 1


def main():
    videos = ['scale', 'translate', 'rotate']

    for video in videos:
        extract_frames(video)


if __name__ == "__main__":
    main()
