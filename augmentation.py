import numpy
import random
import matplotlib.pyplot as plt
import cv2
import copy
import math

class AugBase:
    def __init__(self):
        pass

    # Receives an image and angle, then
    # return a new image with adjusted angle
    def perform_aug(self, img, angle):
        raise TypeError("Base class")

class FlipAug(AugBase):
    def perform_aug(self, img, angle):
        img = numpy.fliplr(img)
        angle = angle * -1

        return img, angle

class BrightnessAug(AugBase):
    def __init__(self, max_delta):
        assert(max_delta <= 1.0 and max_delta > 0)
        self.max_delta = max_delta

    def perform_aug(self, img, angle):
        # To allow the brightness to go either down or up,
        # multiply max delta by 2.0
        # adjust start accordingly
        delta = random.uniform(-self.max_delta, self.max_delta)
        start = 1.0

        # Make img darker
        if (delta < 0):
            start = 1.0 - (self.max_delta)
            delta = 0

        img = cv2.addWeighted(img, start, img, delta, 0.0)
        img = numpy.clip(img, 0, 255)
        img = numpy.array(img, dtype=numpy.uint8)
        return img, angle

class StretchAug(AugBase):
    def __init__(self, max_stretch_ratio):
        self.max_stretch_ratio = max_stretch_ratio

    def perform_aug(self, img, angle):
        org_shape = img.shape
        stretch_ratio = 1.0 + random.uniform(-self.max_stretch_ratio, self.max_stretch_ratio)
        if (stretch_ratio > 1.0) :
            # Enlarge
            img = cv2.resize(img,None,fx=1, fy=stretch_ratio, interpolation = cv2.INTER_CUBIC)
            # Crop
            img = img[img.shape[0] - org_shape[0]:img.shape[0], 0:img.shape[1]]
        elif (stretch_ratio < 1.0):
            # Reduce
            stretched = cv2.resize(img,None,fx=1, fy=stretch_ratio, interpolation = cv2.INTER_CUBIC)
            img = numpy.zeros_like(img)
            x_start = org_shape[1] - stretched.shape[1]
            y_start = org_shape[0] - stretched.shape[0]
            # Replace
            img[y_start:img.shape[0], x_start:img.shape[1]] = stretched
        assert(img.shape == org_shape)
        return img, angle

class ShadowAug(AugBase):
    def perform_aug(self, img, angle):
        num_points = random.randint(3,10)
        points = []

        max_x = img.shape[1]
        max_y = img.shape[0]

        for i in range(num_points - 1):
            points.append((random.randint(0, max_y), random.randint(0, max_x)))

        points.sort()

        shadow_img = copy.copy(img)
        shadow_img = cv2.fillPoly(shadow_img, numpy.array([points]), (0,0,0))
        img = cv2.addWeighted(img, 0.5, shadow_img, 0.5, 0.0)

        return img, angle

class DottedLineAug(AugBase):
    def perform_aug(self, img, angle):
        org_img = copy.copy(img)

        thickness = 5
        color = (255, 255, 255)

        line_segment_length = 20

        max_x = img.shape[1]
        max_y = img.shape[0]

        line_pt_s = (random.randint(int(max_x * 0.25), int(max_x * 0.75)), int(max_y * 0.4))
        line_pt_e = (random.randint(int(max_x * 0.25), int(max_x * 0.75)), random.randint(int(max_y * 0.8), max_y))

        # Eucledean distance between pt a and pt b
        line_total_length = math.sqrt((line_pt_s[0] - line_pt_e[0])**2 + (line_pt_s[1] - line_pt_e[1]) ** 2)
        line_total_length = int(line_total_length)

        # Figure out how much step we should make in x and y
        total_delta = float(abs(line_pt_s[0] - line_pt_e[0]) + abs(line_pt_s[1] - line_pt_e[1]))
        x_ratio = abs((line_pt_s[0] - line_pt_e[0]) / total_delta)
        y_ratio = abs((line_pt_s[1] - line_pt_e[1]) / total_delta)
        x_step = int(x_ratio * line_segment_length)
        y_step = int(y_ratio * line_segment_length)
        x_step *= 1 if line_pt_s[0] < line_pt_e[0] else -1
        y_step *= 1 if line_pt_s[1] < line_pt_e[1] else -1
        
        for i in range(0, line_total_length, line_segment_length):
            line_pt_intermediate = (line_pt_s[0] + x_step, line_pt_s[1] + y_step)
            cv2.line(img,line_pt_s,line_pt_intermediate,color,thickness)
            line_pt_s = (line_pt_intermediate[0] + x_step, line_pt_intermediate[1] + y_step)

        img = cv2.addWeighted(org_img, 0.1, img, 0.9, 0.0)
        return img, angle

class RoadSignAug(AugBase):
    def perform_aug(self, img, angle):
        thickness = 5
        color = (200, 200, 200)
        length = 30

        max_x = img.shape[1]
        max_y = img.shape[0]
        x = random.randint(0, max_x - 1)
        y = random.randint(length, int(max_y * 0.5))
        line_pt_s = (x, y - length)
        line_pt_e = (x, y)

        img = numpy.array(img, dtype=numpy.uint8)
        cv2.line(img,line_pt_s,line_pt_e,color,thickness)
        return img, angle

AUG_COLLECTION = [(0.5, FlipAug()), (1.0, RoadSignAug()), (1.0, ShadowAug()), (1.0, BrightnessAug(0.75)), (1.0, StretchAug(0.4)), (1.0, DottedLineAug())]

def behavior_perform_aug(img, angle):
    for aug_tuple in AUG_COLLECTION:
        aug_prob = aug_tuple[0]
        aug = aug_tuple[1]
        if ((aug_prob * 100) >= random.randint(0, 100)):
            img, angle = aug.perform_aug(img, angle)

    return img, angle

def unit_test(org_img, angle):
    for aug_tuple in AUG_COLLECTION:
        aug = aug_tuple[1]
        img, angle = aug.perform_aug(org_img, angle)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print("angle is %f", angle)
        plt.show()

def main():
    img = cv2.imread('IMG/center_2016_12_01_13_30_48_287.jpg')
    unit_test(img, 10)

if __name__ == '__main__':
    main()
