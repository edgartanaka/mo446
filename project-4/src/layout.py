""""
University of Campinas

MO446 - Computer Vision (Prof. Adin)
Project 4 - Interfaces from sketches

Authors:
Darley Barreto
Edgar Tanaka
Tiago Barros

Scope of this file:
Organisation of the specified patterns.
"""

from scipy.spatial import distance

class Element:
    """Class to store each element/pattern found on the sketch"""
    def __init__(self, type: str, width: float, height: float, x: float, y: float):
        self.type   = type
        self.width  = width
        self.height = height
        self.x      = x
        self.y      = y

class Layout:
    """Class to organise the elements found"""
    def __init__(self, width: float, height: float):
        self.width    = width
        self.height   = height
        self.elements = []

    def addElement(self, element: Element):
        """Method to add a element to the layout"""
        self.elements.append(element)

class TextBlock:
    """Class to organise text blocks (sets of lines)"""
    def __init__(self):
        self.elements = []

    @staticmethod
    def rectangle_union(r1, r2):
        """Method to add two rectangles"""
        x = min(r1[0], r2[0])
        y = min(r1[1], r2[1])
        w = max(r1[0] + r1[2], r2[0] + r2[2]) - x
        h = max(r1[1] + r1[3], r2[1] + r2[3]) - y
        return (x, y, w, h)

    @staticmethod
    def are_lines_together(l1, l2, threshold = 70):
        """Method to check if two lines should be in the same text block"""
        if distance.euclidean((l1[0], l1[1]), (l2[0], l2[1])) < threshold or \
           distance.euclidean((l1[0], l1[1] + l1[3]), (l2[0], l2[1])) < threshold:
            return True
        else:
            return False

    def add(self, bb: tuple):
        """Method to add one line to a text block or create a new one"""
        for i, el in enumerate(self.elements):
            if TextBlock.are_lines_together(el, bb):
                self.elements[i] = TextBlock.rectangle_union(el, bb)
                return

        self.elements.append(bb)
