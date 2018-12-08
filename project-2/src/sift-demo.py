""""
University of Campinas

MO446 - Computer Vision (Prof. Adin)
Project 2 - Augmented Reality

Authors:
Darley Barreto
Edgar Tanaka
Tiago Barros

Scope of this file:
Demonstration of our implementation of Scale-Invariant Feature Transform (SIFT)
"""

import sift
import cv2

def main():

    # List of images to demonstrate our SIFT implementation with
    imageList = [
        "input/frames/rotate/frame0.jpg",
        "input/frames/rotate/frame25.jpg",
        "input/frames/rotate/frame57.jpg",
        "input/frames/rotate/frame142.jpg",
        "input/frames/scale/frame0.jpg",
        "input/frames/scale/frame37.jpg",
        "input/frames/scale/frame55.jpg",
        "input/frames/translate/frame67.jpg",
        "input/frames/translate/frame206.jpg",
        "input/frames/translate/frame258.jpg"
    ]

    # Counter used to number the images when saving them
    counter = 1

    for img in imageList:

        # Scale down the image to speedup the execution
        image = cv2.pyrDown(cv2.imread(img, cv2.IMREAD_COLOR))

        demo = sift.SIFT(image, 2, 1)
        demo.detect()
        demo.drawKeypoints("output/sift-demo/frame{0:02}.png".format(counter))

        counter += 1

if __name__ == '__main__':
    main()
