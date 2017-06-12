import numpy
import random
import matplotlib.pyplot as plt
import cv2
import copy

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
        self.max_delta = max_delta

    def perform_aug(self, img, angle):
        to_add = random.randint(-self.max_delta, self.max_delta)
        img = numpy.add(img, int(to_add))
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

AUG_COLLECTION = [BlackAug(0.6), FlipAug()]

def behavior_perform_aug(aug_prob, img, angle):
    img, angle = AUG_COLLECTION[0].perform_aug(img, angle)
    if ((aug_prob * 100) >= random.randint(0, 100)):
        img, angle = AUG_COLLECTION[1].perform_aug(img, angle)

    return img, angle

def unit_test(img, angle):
    print(img.shape)
    for aug in AUG_COLLECTION:
        img, angle = aug.perform_aug(img, angle)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print("angle is %f", angle)
        plt.show()

def main():
    img = cv2.imread('IMG/center_2016_12_01_13_30_48_287.jpg')
    unit_test(img, 10)

if __name__ == '__main__':
    main()
