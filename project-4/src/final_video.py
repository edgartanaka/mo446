import os
import cv2
import numpy as np

OUTPUT_DIR = 'output'
DIGITS_INDEX = 6
FPS = 5


def build_output_video(video_name):
    screenshots_dir = os.path.join(OUTPUT_DIR, video_name, "screenshots")
    elements_dir = os.path.join(OUTPUT_DIR, video_name, "elements")

    frame1 = cv2.imread(os.path.join(screenshots_dir, 'screenshots_000000.png'), cv2.IMREAD_COLOR)
    height, width, layers = frame1.shape

    # Write all output images into one video
    video = cv2.VideoWriter(os.path.join('output', video_name, video_name + '.avi'),
                            cv2.VideoWriter_fourcc(*"MJPG"),
                            FPS,
                            (width * 2, height))

    frames_count = len([f for f in os.listdir(screenshots_dir) if os.path.isfile(os.path.join(screenshots_dir, f))])

    # going to pick only every 2 frames because gitlab in IC has a limitation of storage :(
    for frame_index in range(0, frames_count - 1, 2):
        screenshot_img = cv2.imread(
            os.path.join(screenshots_dir, 'screenshots_' + str(frame_index).zfill(DIGITS_INDEX) + ".png"))
        elements_img = cv2.imread(
            os.path.join(elements_dir, 'elements_' + str(frame_index).zfill(DIGITS_INDEX) + ".png"))

        double_frame = np.concatenate((elements_img, screenshot_img), axis=1)
        video.write(double_frame)

    cv2.destroyAllWindows()
    video.release()


def main():
    from os import listdir
    from os.path import isfile, join
    video_files = [join('input', f) for f in listdir('input') if isfile(join('input', f))]

    for f in video_files:
        input_path = f

        base = os.path.basename(f)
        video_filename = os.path.splitext(base)[0]  # video name without extension

        if os.path.exists(input_path):
            build_output_video(video_filename)
        else:
            print("Input file \"{0}\" not found!".format(input_path))


if __name__ == "__main__":
    main()
