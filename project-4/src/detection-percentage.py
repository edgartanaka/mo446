""""
University of Campinas

MO446 - Computer Vision (Prof. Adin)
Project 4 - Interfaces from sketches

Authors:
Darley Barreto
Edgar Tanaka
Tiago Barros

Scope of this file:
Detection and organisation of the specified patterns.

Based on shape detection by Adrian Rosebrock:
https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
"""

from layout import TextBlock
from layout import Element
from layout import Layout
import numpy as np
import imutils
import cv2
import os

OUTPUT_DIR = "output"
INPUT_FILE = "easy1.mp4"

def draw_output(image, element, bounding, bounding_boxes, min_x, min_y, max_x, max_y):
        """Method to draw the detection results and update max/min values"""

        cv2.rectangle(image,
                      (bounding[0], bounding[1]),
                      (bounding[0] + bounding[2], bounding[1] + bounding[3]),
                      (0, 0, 255),
                      2)

        cX = int(bounding[0] + 0.5 * bounding[2])
        cY = int(bounding[1] + 0.5 * bounding[3])

        cv2.putText(image, element, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 0, 0), 2)

        bounding_boxes.append(Element(element,
                                      bounding[2],
                                      bounding[3],
                                      bounding[0],
                                      bounding[1]))

        min_x = min(min_x, bounding[0])
        min_y = min(min_y, bounding[1])
        max_x = max(max_x, bounding[0] + bounding[2])
        max_y = max(max_y, bounding[1] + bounding[3])

        return min_x, min_y, max_x, max_y

def element_detect(c):
    """Method to identify a element"""

    # Initialize the element name and approximate the contour
    element = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    bounding = None
    
    if len(approx) == 4:
        # Compute the bounding box of the contour
        bounding = cv2.boundingRect(approx)

        element = "image"
    
    elif len(approx) == 2:
        area = cv2.contourArea(c)
        magic_ratio = area / peri
        bounding = cv2.boundingRect(approx)

        # A header will have a big magic ratio, otherwise, it is a text
        element = "header" if magic_ratio >= 2.0 else "text"
        
    # return the name of the element
    return element, bounding

def find_elements(cap):
    """Method to read the video frames and process them"""
    
    min_tresh = 0
    max_tresh = 70

    def on_min_tresh(val):
        nonlocal min_tresh
        min_tresh = val
    
    def on_max_tresh(val):
        nonlocal max_tresh
        max_tresh = val

    # Creates the output directory, if needed
    basename = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    output_directory = os.path.join(OUTPUT_DIR, basename)
    if not os.path.exists(output_directory):
        try:
            os.makedirs(output_directory)
        except OSError:
            print("Creation of the directory \"{0}\" failed!".format(output_directory))

    # Creates output directory for edges, if needed
    if not os.path.exists(os.path.join(output_directory, "edges")):
        try:
            os.makedirs(os.path.join(output_directory, "edges"))
        except OSError:
            print("Creation of the directory for edges failed!")

    # Creates output directory for elements, if needed
    if not os.path.exists(os.path.join(output_directory, "elements")):
        try:
            os.makedirs(os.path.join(output_directory, "elements"))
        except OSError:
            print("Creation of the directory for elements failed!")

    counter = 0

    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        
        frame_threshold, edges, bounding_boxes, min_x, min_y, max_x, max_y = \
                detect(frame,min_tresh,max_tresh)

        cv2.imwrite(os.path.join(output_directory, "edges",
            "edges_" + str(counter) + ".png"), edges)
        cv2.imwrite(os.path.join(output_directory, "elements",
            "elements_" + str(counter) + ".png"), frame_threshold)

        if len(bounding_boxes) > 0:
            layout_width  = max_x - min_x
            layout_height = max_y - min_y

            layout = Layout(layout_width, layout_height)

            for element in bounding_boxes:

                # Gets the percentage value of the X coordinate
                x = 100 * ((element.x - min_x) / layout_width)

                # Considers only five values to compensate for the noise
                x = 20 * round(x / 20)

                # Gets the percentage value of the Y coordinate
                y = 100 * ((element.y - min_y) / layout_height)

                # Considers only five values to compensate for the noise
                y = 20 * round(y / 20)

                # Gets the percentage value of the width
                width = 100 * (element.width / layout_width)

                # Considers only five values to compensate for the noise
                width = 20 * round(width / 20)

                # Gets the percentage value of the height
                height = 100 * (element.height / layout_height)

                # Considers only five values to compensate for the noise
                height = 20 * round(height / 20)

                # Updates the values of element to be percentages
                element.width  = width
                element.height = height
                element.x      = x
                element.y      = y

                layout.addElement(element)

        counter += 1

def detect(image, min_tresh, max_tresh):
    """Method to detect the elements/patterns in the current frame"""

    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, min_tresh, max_tresh,apertureSize=3)
    
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[1]

    bounding_boxes = []
    text_blocks = TextBlock()

    min_x = image.shape[1]
    min_y = image.shape[0]
    max_x = 0
    max_y = 0
    
    for c in cnts:
        # Compute the center of the contour, then detect the name of the
        # element using only the contour
        M = cv2.moments(c)
        if not (M["m10"] and M["m00"]): continue
            
        element, bounding = element_detect(c)
        
        if element == "unidentified": continue
            
        # Multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the element on the image
        
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")

        bounding = tuple(round(i * ratio) for i in bounding)

        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

        if element == "text":
            text_blocks.add(bounding)
            continue

        min_x, min_y, max_x, max_y = draw_output(image,
                                                 element,
                                                 bounding,
                                                 bounding_boxes,
                                                 min_x,
                                                 min_y,
                                                 max_x,
                                                 max_y)

    for el in text_blocks.elements:
        min_x, min_y, max_x, max_y = draw_output(image,
                                                 "text",
                                                 el,
                                                 bounding_boxes,
                                                 min_x,
                                                 min_y,
                                                 max_x,
                                                 max_y)

    return image, edges, bounding_boxes, min_x, min_y, max_x, max_y

def main():
    input_path = os.path.join("input", INPUT_FILE)

    if os.path.exists(input_path):
        cap = cv2.VideoCapture(input_path)
        find_elements(cap)
    else:
        print("Input file \"{0}\" not found!".format(input_path))

if __name__ == "__main__":
    main()
