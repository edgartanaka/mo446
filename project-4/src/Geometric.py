import numpy as np
import cv2
import imutils


def shape_detect(c):
    # initialize the shape name and approximate the contour
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    bounding = None
    
    if len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        bounding = cv2.boundingRect(approx)
        _, _, w, h = bounding
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    
    elif len(approx) == 2:
        bounding = cv2.boundingRect(approx)
        shape = "line"
        
    # return the name of the shape
    return shape, bounding

def find_shapes(cap):
    
    min_tresh = 0
    max_tresh = 70

    def on_min_tresh(val):
        nonlocal min_tresh
        min_tresh = val
    
    def on_max_tresh(val):
        nonlocal max_tresh
        max_tresh = val
       
    window_capture_name = 'Video Capture'
    window_detection_name = 'Object Detection'
          
#     cv2.namedWindow(window_capture_name)
    cv2.namedWindow(window_detection_name)
    cv2.namedWindow("Edges")
    
    cv2.createTrackbar("Min tresh", 'Edges' , 0, 255, on_min_tresh)
    cv2.createTrackbar("Max tresh", 'Edges' , 0, 255, on_max_tresh)

    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        
        frame_threshold,edges = detect(frame,min_tresh,max_tresh)

#         cv2.imshow(window_capture_name, frame)
        cv2.imshow("Edges", edges)
        cv2.imshow(window_detection_name, frame_threshold)

        key = cv2.waitKey(30)
        if key == ord('q') or key == 27:
            break

def detect(image,min_tresh,max_tresh):    
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, min_tresh, max_tresh,apertureSize=3)
    
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        if not (M["m10"] and M["m00"]): continue
            
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape, bouding = shape_detect(c)
        
        if shape == 'unidentified': continue
            
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 0, 0), 2)
        
    return image,edges


cap = cv2.VideoCapture('../input/hard.mp4')
find_shapes(cap)

