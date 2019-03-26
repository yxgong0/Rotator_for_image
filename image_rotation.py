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


class Rotator:
    def __init__(self, image, points=[], rects=[], np_rotated_rects=np.zeros((1,5)), cv_rotated_rects=[],
                 quadrilaterals=[], polygons=[], expand_edge=True):
        self.image = image
        self.points = points
        self.rects = rects
        self.np_rotated_rects = np_rotated_rects
        self.cv_rotated_rects = cv_rotated_rects
        self.quadrilaterals = quadrilaterals
        self.polygons = polygons
        self.rotation_angle = (-15, 15)
        self.expand_edge=expand_edge

        self.points_flag = False
        self.rects_flag = False
        self.np_rotated_rects_flag = False
        self.cv_rotated_rects_flag = False
        self.quadrilaterals_flag = False
        self.polygons_flag = False

        self.angle = 0
        self.radian = 0
        self.width_increased = 0
        self.height_increased = 0
        self.new_w = self.image.shape[1]
        self.new_h = self.image.shape[0]

        self.check_parameters()

        self.results = {'image':None, 'points':None, 'rects':None, 'np_rotated_rects':None, 'cv_rotated_rects':None,
                        'quadrilaterals':None, 'polygons':None}

    def check_parameters(self):
        assert isinstance(self.image, np.ndarray)
        assert isinstance(self.expand_edge, bool)
        assert isinstance(self.points, list) or isinstance(self.points, tuple)
        assert isinstance(self.rects, list) or isinstance(self.rects, tuple)
        assert isinstance(self.np_rotated_rects, np.ndarray) and self.np_rotated_rects.shape.__len__() == 2
        assert isinstance(self.cv_rotated_rects, tuple) or isinstance(self.cv_rotated_rects, list)
        assert isinstance(self.quadrilaterals, list) or isinstance(self.quadrilaterals, tuple)
        assert isinstance(self.polygons, list) or isinstance(self.polygons, tuple)
        assert (isinstance(self.rotation_angle, tuple) and self.rotation_angle.__len__() == 2) \
               or isinstance(self.rotation_angle, int)
        if self.points.__len__() > 0:
            self.points_flag = True
            for point in self.points:
                assert isinstance(point, tuple) and point.__len__() == 2
        if self.rects.__len__() > 0:
            self.rects_flag = True
            for rect in rects:
                assert isinstance(rect, tuple) and rect.__len__() == 4
        if self.np_rotated_rects.nonzero().__len__() != 0:
            self.np_rotated_rects_flag = True
            for r in range(self.np_rotated_rects.shape[0]):
                line = self.np_rotated_rects[r, :]
                assert line.__len__() == 5
        if self.cv_rotated_rects.__len__() != 0:
            self.cv_rotated_rects_flag = True
            for rect in self.cv_rotated_rects:
                assert isinstance(rect, tuple) and rect.__len__() == 3
                assert rect[0].__len__() == 2 and rect[1].__len__() == 2
                assert isinstance(rect[2], float)
        if self.quadrilaterals.__len__() != 0:
            self.quadrilaterals_flag = True
            for quadrilateral in self.quadrilaterals:
                assert (isinstance(quadrilateral, tuple) or isinstance(quadrilateral, list))\
                       and quadrilateral.__len__() == 8
        if self.polygons.__len__() != 0:
            self.polygons_flag = True
            for polygon in polygons:
                assert isinstance(polygon, tuple) or isinstance(polygon, list)
                for pt in polygon:
                    assert isinstance(pt, tuple) and pt.__len__() == 2

    # A method to calculate the coordinate of the points after rotating
    def rotate_point(self, point):
        assert isinstance(point, tuple) and point.__len__() == 2

        point_expanded = (point[0] + self.width_increased / 2., point[1] + self.height_increased / 2.)
        point_translated = (point_expanded[0] - self.new_w / 2., self.new_h / 2. - point_expanded[1])
        point_rotated = (point_translated[0] * math.cos(self.radian) - point_translated[1] * math.sin(self.radian),
                         point_translated[0] * math.sin(self.radian) + point_translated[1] * math.cos(self.radian))
        point_restored = (int(round(point_rotated[0] + self.new_w / 2.)),
                          int(round(-1 * point_rotated[1] + self.new_h / 2.)))

        return point_restored

    def rotate(self, rotation_angle=None):
        if rotation_angle is not None:
            self.rotation_angle = rotation_angle
        self.check_parameters()

        # Generate a random angle with the given range for data augmentation if the input is not an integer
        if isinstance(self.rotation_angle, int):
            angle = self.rotation_angle
        else:
            angle = random.randint(self.rotation_angle[0], self.rotation_angle[1])
        # Transform the rotation angle to [0, 360) and calculate the corresponding radian
        while angle < 0:
            angle += 360
        angle = angle % 360
        radian = angle * 3.1415927 / 180.
        self.angle = angle
        self.radian = radian

        # Calculate the new width and height after rotation
        (h, w) = self.image.shape[:2]
        if self.expand_edge:
            new_w = round(abs(h * math.sin(radian)) + abs(w * math.cos(radian)))
            new_h = round(abs(w * math.sin(radian)) + abs(h * math.cos(radian)))
        else:
            new_w = w
            new_h = h
        height_increment = int(round(new_h - h))
        width_increment = int(round(new_w - w))
        self.new_w = new_w
        self.new_h = new_h
        self.height_increased = height_increment
        self.width_increased = width_increment

        # Extend the edges of the initial image according to the new width and height
        image_expanded = np.zeros((int(new_h), int(new_w), 3), np.uint8)
        image_expanded[height_increment // 2:height_increment // 2 + h, width_increment // 2:width_increment // 2 + w]\
            = self.image

        # Rotate the image
        M = cv2.getRotationMatrix2D((new_w / 2., new_h / 2.), angle, 1.0)
        image_rotated = cv2.warpAffine(image_expanded, M, (int(new_w), int(new_h)))
        self.results['image'] = image_rotated

        if self.points_flag:
            # Calculate the new coordinates of the input points
            new_points = []
            for point in points:
                new_point = self.rotate_point(point)
                new_points.append(new_point)
            self.results['points'] = new_points

        if self.rects_flag:
            # Produce the rotated rect from rect
            new_rects = []
            for rect in rects:
                center = (rect[0] + rect[2] / 2., rect[1] + rect[3] / 2.)
                new_center = self.rotate_point(center)
                tl_point = (rect[0], rect[1])
                new_tl_point = self.rotate_point(tl_point)
                tr_point = (rect[0] + rect[2], rect[1])
                new_tr_point = self.rotate_point(tr_point)
                br_point = (rect[0] + rect[2], rect[1] + rect[3])
                new_br_point = self.rotate_point(br_point)
                bl_point = (rect[0], rect[1] + rect[3])
                new_bl_point = self.rotate_point(bl_point)
                new_rect = RotatedRect(new_center, rect[2], rect[3], angle,
                                       [new_tl_point, new_tr_point, new_br_point, new_bl_point])
                new_rects.append(new_rect)
            self.results['rects'] = new_rects

        if self.np_rotated_rects_flag:
            # Produce the rotated rect from np_rotated_rect
            new_np_rotated_rects = np.zeros(self.np_rotated_rects.shape)
            for r in range(self.np_rotated_rects.shape[0]):
                x_min = self.np_rotated_rects[r, 0]
                y_min = self.np_rotated_rects[r, 1]
                x_max = self.np_rotated_rects[r, 2]
                y_max = self.np_rotated_rects[r, 3]
                theta = self.np_rotated_rects[r, 4]
                tl_point = (x_min, y_min)
                br_point = (x_max, y_max)
                new_tl_point = self.rotate_point(tl_point)
                new_br_point = self.rotate_point(br_point)
                new_theta = theta - angle
                new_np_rotated_rects[r, 0] = new_tl_point[0]
                new_np_rotated_rects[r, 1] = new_tl_point[1]
                new_np_rotated_rects[r, 2] = new_br_point[0]
                new_np_rotated_rects[r, 3] = new_br_point[1]
                new_np_rotated_rects[r, 4] = new_theta

                self.results['np_rotated_rects'] = new_np_rotated_rects

        if self.cv_rotated_rects_flag:
            # Produce the rotated rect from cv_rotated_rect
            new_cv_rotated_rects = []
            for cv_rotated_rect in self.cv_rotated_rects:
                box_points = cv2.boxPoints(cv_rotated_rect)
                pt1 = (box_points[0][0], box_points[0][1])
                pt2 = (box_points[1][0], box_points[1][1])
                pt3 = (box_points[2][0], box_points[2][1])
                pt4 = (box_points[3][0], box_points[3][1])
                new_pt1 = self.rotate_point(pt1)
                new_pt2 = self.rotate_point(pt2)
                new_pt3 = self.rotate_point(pt3)
                new_pt4 = self.rotate_point(pt4)
                pts = [new_pt1, new_pt2, new_pt3, new_pt4]
                new_cv_rotated_rect = cv2.minAreaRect(np.array(pts))
                new_cv_rotated_rects.append(new_cv_rotated_rect)
            self.results['cv_rotated_rects'] = new_cv_rotated_rects

        if self.quadrilaterals_flag:
            # Produce the rotated quadrilaterals from input ones
            new_quadrilaterals = []
            for quadrilateral in self.quadrilaterals:
                point1 = (quadrilateral[0], quadrilateral[1])
                point2 = (quadrilateral[2], quadrilateral[3])
                point3 = (quadrilateral[4], quadrilateral[5])
                point4 = (quadrilateral[6], quadrilateral[7])
                new_point1 = self.rotate_point(point1)
                new_point2 = self.rotate_point(point2)
                new_point3 = self.rotate_point(point3)
                new_point4 = self.rotate_point(point4)
                new_quadrilateral = (new_point1[0], new_point1[1], new_point2[0], new_point2[1],
                                     new_point3[0], new_point3[1], new_point4[0], new_point4[1],)
                new_quadrilaterals.append(new_quadrilateral)
            self.results['quadrilaterals'] = new_quadrilaterals

        if self.polygons_flag:
            # Produce the rotated polygons from input ones
            new_polygons = []
            for polygon in polygons:
                new_polygon = []
                for pt in polygon:
                    new_pt = self.rotate_point(pt)
                    new_polygon.append(new_pt)
                new_polygons.append(tuple(new_polygon))
            self.results['polygons'] = new_polygons

        return self.results


if __name__ == '__main__':
    # Read image to be rotated
    image = cv2.imread('D:/test.jpg')
    image_ = image.copy()
    # Annotations formatted as points
    points = [(113, 127), (156, 127), (156, 170), (113, 170), (179, 127), (224, 127), (224, 171), (179, 171)]
    for point in points:
        cv2.circle(image_, point, 1, (0, 0, 255))
    # Annotations formatted as rects [x, y, width, height]
    rects = [(117, 188, 108, 57)]
    cv2.rectangle(image_, (rects[0][0], rects[0][1]), (rects[0][0] + rects[0][2], rects[0][1] + rects[0][3]),
                  (0, 255, 0))
    # Annotations of rotated rectangles formatted as numpy.array
    np_rotated_rects = np.zeros((1, 5))
    np_rotated_rects[0, 0] = 154
    np_rotated_rects[0, 1] = 66
    np_rotated_rects[0, 2] = 186
    np_rotated_rects[0, 3] = 90
    np_rotated_rects[0, 4] = 0
    for r in range(np_rotated_rects.shape[1]):
        x_min = int(np_rotated_rects[0, 0])
        y_min = int(np_rotated_rects[0, 1])
        x_max = int(np_rotated_rects[0, 2])
        y_max = int(np_rotated_rects[0, 3])
        theta = int(np_rotated_rects[0, 4])
        cv2.circle(image_, (x_min, y_min), 1, (255, 0, 0))
        cv2.circle(image_, (x_max, y_max), 1, (255, 0, 0))
    # Annotations of rotated rectangles formatted as opencv
    pts = [(133, 142), (136, 141), (141, 147), (139, 148), (132, 151), (137, 154), (128, 150), (141, 154)]
    cv_rotated_rect = cv2.minAreaRect(np.array(pts))
    cv_rotated_rects = [cv_rotated_rect]
    box = cv2.boxPoints(cv_rotated_rect)
    for i in range(box.__len__()):
        cv2.line(image_, (int(box[i][0]), int(box[i][1])),
                 (int(box[(i+1) % 4][0]), int(box[(i+1) % 4][1])), (255, 128, 255))
    # Annotations of quadrilaterals
    quadrilaterals = [(163, 276, 206, 300, 217, 337, 143, 316)]
    for n in range(quadrilaterals.__len__()):
        for i in range(0, 7, 2):
            pt1 = (quadrilaterals[n][i], quadrilaterals[n][i+1])
            pt2 = (quadrilaterals[n][(i+2) % 8], quadrilaterals[n][(i+3) % 8])
            cv2.line(image_, pt1, pt2, (255, 255, 0))
    # Annotations of polygons
    polygons = [((271, 286), (296, 305), (286, 337), (255, 337), (245, 305)),
                ((231, 266), (243, 291), (220, 291))]
    for n in range(polygons.__len__()):
        for i in range(polygons[n].__len__()):
            cv2.line(image_, polygons[n][i], polygons[n][(i+1) % polygons[n].__len__()], (255, 0, 255))

    cv2.imshow('initial_image', image_)

    # Define the rotater and rotate the image
    rotator = Rotator(image, points=points, rects=rects, np_rotated_rects=np_rotated_rects,
                      cv_rotated_rects=cv_rotated_rects, quadrilaterals=quadrilaterals, polygons=polygons,
                      expand_edge=True)
    results = rotator.rotate(15)

    image_rotated = results['image']
    # Draw rotated annotations of points
    new_points = results['points']
    for new_point in new_points:
        cv2.circle(image_rotated, new_point, 1, (0, 0, 255))
    # Draw rotated annotations of rectangles
    new_rects = results['rects']
    for new_rect in new_rects:
        box_points = new_rect.get_box_points()
        for i in range(box_points.__len__()):
            cv2.line(image_rotated, box_points[i], box_points[(i + 1) % 4], (0, 255, 0))
    # Draw rotated annotations of rectangles formatted as numpy.array
    new_np_rotated_rects = results['np_rotated_rects']
    for r in range(new_np_rotated_rects.shape[1]):
        x_min = int(new_np_rotated_rects[0, 0])
        y_min = int(new_np_rotated_rects[0, 1])
        x_max = int(new_np_rotated_rects[0, 2])
        y_max = int(new_np_rotated_rects[0, 3])
        theta = int(new_np_rotated_rects[0, 4])
        cv2.circle(image_rotated, (x_min, y_min), 1, (255, 0, 0))
        cv2.circle(image_rotated, (x_max, y_max), 1, (255, 0, 0))
    # Draw rotated annotations of rectangles formatted as opencv
    cv_rotated_rects = results['cv_rotated_rects']
    cv_rotated_rect = cv_rotated_rects[0]
    box = cv2.boxPoints(cv_rotated_rect)
    for i in range(cv_rotated_rect.__len__()):
        cv2.line(image_rotated, (int(box[i][0]),int(box[i][1])),
                 (int(box[(i+1) % 4][0]), int(box[(i+1) % 4][1])), (255, 128, 255))
    # Draw rotated annotations of quadrilaterals
    new_quadrilaterals = results['quadrilaterals']
    for n in range(new_quadrilaterals.__len__()):
        for i in range(0, 7, 2):
            pt1 = (new_quadrilaterals[n][i], new_quadrilaterals[n][i+1])
            pt2 = (new_quadrilaterals[n][(i+2) % 8], new_quadrilaterals[n][(i+3) % 8])
            cv2.line(image_rotated, pt1, pt2, (255,255,0))
    # Draw rotated annotations of polygons
    new_polygons = results['polygons']
    for n in range(new_polygons.__len__()):
        for i in range(new_polygons[n].__len__()):
            cv2.line(image_rotated, new_polygons[n][i], new_polygons[n][(i+1) % new_polygons[n].__len__()], (255, 0, 255))



    # Show the results
    cv2.imshow('rotated_image', image_rotated)
    cv2.waitKey(0)
