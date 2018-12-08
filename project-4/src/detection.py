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
import imutils
import cv2
import os
from html import HtmlBuilder
import shutil

OUTPUT_DIR = "output"

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
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)
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
        element = "header" if magic_ratio >= 3.0 else "text"
        
    # return the name of the element
    return element, bounding

def find_elements(cap, basename):
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

    # Creates output directory for html, if needed
    if not os.path.exists(os.path.join(output_directory, "html")):
        try:
            os.makedirs(os.path.join(output_directory, "html"))
        except OSError:
            print("Creation of the directory for html failed!")

    # Creates output directory for screenshots, if needed
    if not os.path.exists(os.path.join(output_directory, "screenshots")):
        try:
            os.makedirs(os.path.join(output_directory, "screenshots"))
        except OSError:
            print("Creation of the directory for screenshots failed!")

    counter = 0
    while True:
        ret, frame = cap.read()

        if frame is None:
            break

        frame_threshold, edges, bounding_boxes, min_x, min_y, max_x, max_y = \
                detect(frame,min_tresh,max_tresh)

        cv2.imwrite(os.path.join(output_directory, "edges",
            "edges_" + str(counter).zfill(6) + ".png"), edges)
        cv2.imwrite(os.path.join(output_directory, "elements",
            "elements_" + str(counter).zfill(6) + ".png"), frame_threshold)

        html_builder = HtmlBuilder()
        if len(bounding_boxes) > 0:
            layout_width  = max_x - min_x
            layout_height = max_y - min_y

            layout = Layout(layout_width, layout_height)

            for element in bounding_boxes:
                layout.addElement(element)

            # HTML compose
            html_code = html_builder.get_html(layout)
        else:
            html_code = html_builder.get_blank_html()

        # Save HTML file
        html_file = os.path.join(output_directory, "html", "html_" + str(counter).zfill(6) + ".html")
        f = open(html_file, "w")
        f.write(html_code)
        f.close()

        # Take screenshot
        html_file = os.path.abspath(html_file)
        os.system(
            "google-chrome-stable --headless --disable-gpu --screenshot --window-size=404,720 --no-sandbox file://" + html_file)

        # copy local file screenshot to the screenshots directory
        screenshot_file = os.path.join(output_directory, "screenshots", "screenshots_" + str(counter).zfill(6) + ".png")
        shutil.copyfile('screenshot.png', screenshot_file)

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
    elements = []
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

        elements.append(element)

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
    from os import listdir
    from os.path import isfile, join
    video_files = [join('input', f) for f in listdir('input') if isfile(join('input', f))]

    for f in video_files:
        base = os.path.basename(f)
        basename = os.path.splitext(base)[0] # video name without extension

        if os.path.exists(f):
            cap = cv2.VideoCapture(f)
            find_elements(cap, basename)
        else:
            print("Input file \"{0}\" not found!".format(f))

if __name__ == "__main__":
    main()
