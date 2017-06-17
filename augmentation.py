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

class BlackAug(AugBase):
    def __init__(self, length):
        self.length = length

    def perform_aug(self, img, angle):
        img_w = img.shape[1]
        left_or_right = random.randint(0, 2)
        start = 0
        end = 0

        if (left_or_right == 0):
            start = 0
            end = int(img_w * self.length)
        else:
            start = int(img_w * (1 - self.length))
            end = img_w

        img = copy.copy(img)
        img[:, start:end, :] = 0
        return img, angle

class FlipAug(AugBase):
    def perform_aug(self, img, angle):
        img = numpy.fliplr(img)
        angle = angle * -1

        return img, angle

class BrightnessAug(AugBase):
    def __init__(self, max_delta):
        self.max_delta = int(max_delta * 100)

    def perform_aug(self, img, angle):
        start = 100 - self.max_delta
        start /= 100.0
        delta = random.randint(0, self.max_delta * 2)
        delta /= 100.0

        img = cv2.addWeighted(img, start, img, delta, 0.0)
        img = numpy.clip(img, 0, 255)
        img = numpy.array(img, dtype=numpy.uint8)
        return img, angle

class StretchAug(AugBase):
    def __init__(self, stretch_ratio):
        self.stretch_ratio = stretch_ratio

    def perform_aug(self, img, angle):
        org_shape = img.shape
        img = cv2.resize(img,None,fx=1, fy=self.stretch_ratio, interpolation = cv2.INTER_CUBIC)
        # Crop
        img = img[img.shape[0] - org_shape[0]:img.shape[0], 0:img.shape[1]]
        return img, angle

class ShadowAug(AugBase):
    def __init__(self, step_range):
        self.step_range = step_range

    def get_point(self, center_x, center_y, max_x, max_y):
        x = None
        y = None

        while x is None or y is None:
            x = random.randint(center_x - self.step_range, center_x + self.step_range)
            y = random.randint(center_y - self.step_range, center_y + self.step_range)
            # Instead of cropping, regenerate if over the range
            # to avoid too many points being at the edge of the img

            if x < 0 or x >= max_x:
                x = None
            if y < 0 or y >= max_y:
                y = None

        return (x, y)

    def perform_aug(self, img, angle):
        num_points = random.randint(3, 10)
        points = []

        max_x = img.shape[1]
        center_x = random.randint(self.step_range, max_x - self.step_range)
        max_y = img.shape[0]
        center_y = random.randint(self.step_range, max_y - self.step_range)

        for i in range(num_points - 1):
            points.append(self.get_point(center_x, center_y, max_x, max_y))

        points.sort()

        shadow_img = copy.copy(img)
        shadow_img = cv2.fillPoly(shadow_img, numpy.array([points]), (0,0,0))
        img = cv2.addWeighted(img, 0.5, shadow_img, 0.5, 0.0)

        return img, angle

AUG_COLLECTION = [BlackAug(0.5), FlipAug(), ShadowAug(50), BrightnessAug(0.25), StretchAug(1.3)]

def behavior_perform_aug(aug_prob, img, angle):
    for aug in AUG_COLLECTION:
        if ((aug_prob * 100) >= random.randint(0, 100)):
            img, angle = aug.perform_aug(img, angle)

    return img, angle

def unit_test(org_img, angle):
    for aug in AUG_COLLECTION:
        img, angle = aug.perform_aug(org_img, angle)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print("angle is %f", angle)
        plt.show()

def main():
    img = cv2.imread('IMG/center_2016_12_01_13_30_48_287.jpg')
    unit_test(img, 10)

if __name__ == '__main__':
    main()
