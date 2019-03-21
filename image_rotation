import numpy as np
import cv2
import math
import random


# Definition of rotated rectangles, where box_points is a list of the four vertexes of the rectangle
# The range of the angle is [0, 360)
class RotatedRect:
    def __init__(self, center, width, height, angle, box_points):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.box_points = box_points

    def get_box_points(self):
        return self.box_points


# A method to calculate the coordinate of the points after rotating
def rotate_point(point, radian, width_increased, height_increased, new_w, new_h):
    assert isinstance(point, tuple) and point.__len__() == 2

    point_expanded = (point[0] + width_increased / 2., point[1] + height_increased / 2.)
    point_translated = (point_expanded[0] - new_w / 2., new_h / 2. - point_expanded[1])
    point_rotated = (point_translated[0] * math.cos(radian) - point_translated[1] * math.sin(radian),
                     point_translated[0] * math.sin(radian) + point_translated[1] * math.cos(radian))
    point_restored = (math.ceil(point_rotated[0] + new_w / 2.), math.ceil(-1 * point_rotated[1] + new_h / 2.))

    return point_restored


def random_rotation(image, points=[], rects=[], rotation_angle=(-15, 15)):
    # To validate the type of each parameter
    assert isinstance(image, np.ndarray)
    assert isinstance(points, list)
    assert isinstance(rects, list)
    assert (isinstance(rotation_angle, tuple) and rotation_angle.__len__() == 2) or isinstance(rotation_angle, int)
    if points.__len__() > 0:
        for point in points:
            assert isinstance(point, tuple) and point.__len__() == 2
    if rects.__len__() > 0:
        for rect in rects:
            assert isinstance(rect, tuple) and rect.__len__() == 4

    # Generate a random angle with the given range for data augmentation if the input is not an integer
    if isinstance(rotation_angle, int):
        angle = rotation_angle
    else:
        angle = random.randint(rotation_angle[0], rotation_angle[1])
    # Transform the rotation angle to [0, 360) and calculate the corresponding radian
    while angle < 0:
        angle += 360
    angle = angle % 360
    radian = angle * 3.1416 / 180.

    # Calculate the new width and height after rotation
    (h, w) = image.shape[:2]
    new_w = math.ceil(abs(h * math.sin(radian)) + abs(w * math.cos(radian)))
    new_h = math.ceil(abs(w * math.sin(radian)) + abs(h * math.cos(radian)))
    height_increment = math.ceil(new_h - h)
    width_increment = math.ceil(new_w - w)

    # Extend the edges of the initial image according to the new width and height
    image_expanded = np.zeros((new_h, new_w, 3), np.uint8)
    for s in range(new_h):
        for t in range(new_w):
            if height_increment // 2 <= s < height_increment // 2 + h and\
                    width_increment // 2 <= t < width_increment // 2 + w:
                image_expanded[s, t][0] = image[s - height_increment // 2, t - width_increment // 2][0]
                image_expanded[s, t][1] = image[s - height_increment // 2, t - width_increment // 2][1]
                image_expanded[s, t][2] = image[s - height_increment // 2, t - width_increment // 2][2]

    # Rotate the image
    M = cv2.getRotationMatrix2D((new_h / 2., new_w / 2.), angle, 1.0)
    image_rotated = cv2.warpAffine(image_expanded, M, (new_w, new_h))

    # Calculate the new coordinates of the input points
    new_points = []
    for point in points:
        new_point = rotate_point(point, radian, width_increment, height_increment, new_w, new_h)
        new_points.append(new_point)

    # Produce the rotated rect
    new_rects = []
    for rect in rects:
        center = (rect[0] + rect[2] / 2., rect[1] + rect[3] / 2.)
        new_center = rotate_point(center, radian, width_increment, height_increment, new_w, new_h)
        tl_point = (rect[0], rect[1])
        new_tl_point = rotate_point(tl_point, radian, width_increment, height_increment, new_w, new_h)
        tr_point = (rect[0] + rect[2], rect[1])
        new_tr_point = rotate_point(tr_point, radian, width_increment, height_increment, new_w, new_h)
        br_point = (rect[0] + rect[2], rect[1] + rect[3])
        new_br_point = rotate_point(br_point, radian, width_increment, height_increment, new_w, new_h)
        bl_point = (rect[0], rect[1] + rect[3])
        new_bl_point = rotate_point(bl_point, radian, width_increment, height_increment, new_w, new_h)
        new_rect = RotatedRect(new_center, rect[2], rect[3], angle,
                               [new_tl_point, new_tr_point, new_br_point, new_bl_point])
        new_rects.append(new_rect)

    return image_rotated, new_points, new_rects


if __name__ == '__main__':
    image = cv2.imread('D:/test.jpg')
    image_ = image.copy()
    points = [(113, 127), (156, 127), (156, 170), (113, 170), (179, 127), (224, 127), (224, 171), (179, 171)]
    for point in points:
        cv2.circle(image_, point, 1, (0, 0, 255))
    rects = [(117, 188, 108, 57)]
    cv2.rectangle(image_, (rects[0][0], rects[0][1]), (rects[0][0] + rects[0][2], rects[0][1] + rects[0][3]),
                  (0, 0, 255))
    cv2.imshow('initial_image', image_)

    image_rotated, new_points, new_rects = random_rotation(image, points, rects)
    for new_point in new_points:
        cv2.circle(image_rotated, new_point, 1, (0, 0, 255))
    for new_rect in new_rects:
        box_points = new_rect.get_box_points()
        for i in range(box_points.__len__()):
            cv2.line(image_rotated, box_points[i], box_points[(i + 1) % 4], (0, 0, 255))
    cv2.imshow('rotated_image', image_rotated)
    cv2.waitKey(0)
