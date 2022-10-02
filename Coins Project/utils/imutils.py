import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt



def translate(image, x, y):
    """It move the image in x and y direction"""
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv.warpAffine(image, M, (image.shape[1], image.shape[0]))

def rotation(image, angle, center = None, scale = 1.0):
    """It rotates the image with default center at the center of the image by accepting the angle to which it will rotate image"""
    (h, w) = image.shape[:2]
    if center == None:
        center = (h // 2, w // 2)
    
    M = cv.getRotationMatrix2D(center, angle, scale)
    return cv.warpAffine(image, M, (w, h))

def resize(image, width = None, height = None, interpolation=cv.INTER_AREA):
    """It resize the image with respect to width or height"""
    dimension = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / h
        dimension = (int(w * r), height)
    else:
        r = width / w
        dimension = (width, int(h * r))

    return cv.resize(image, dimension, interpolation)


def resizeRatio(image, width=None, height=None, interpolation=cv.INTER_NEAREST):
    if width is None and height is None:
        return image
    if width is None:
        ratio = image.shape[1] / image.shape[0]
        dimension = (int((ratio * height)), height)
    else:
        ratio = image.shape[0] / image.shape[1]
        dimension = (width, int((ratio * width)))
    
   
    return cv.resize(image, dimension, interpolation)   

def circleMask(image, radius=None):
    """It create the circular mask and return the mask and masked image as a tuple"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    mask = np.zeros(image.shape[:2], dtype='uint8')
    if radius is None: 
        cv.circle(mask, center, (h + w) // 4, 255, -1)
    else: 
        cv.circle(mask, center, radius, 255, -1)

    return (mask, cv.bitwise_and(image, image, mask=mask))

def circleMask(image, center, radius=None):
    """It create the circular mask and return the mask and masked image as a tuple"""
    (h, w) = image.shape[:2]

    mask = np.zeros(image.shape[:2], dtype='uint8')
    if radius is None: 
        cv.circle(mask, center, (h + w) // 4, 255, -1)
    else: 
        cv.circle(mask, center, radius, 255, -1)

    return (mask, cv.bitwise_and(image, image, mask=mask))

def rectangleMask(image, percentage=None):
    """It create a rectangular mask and return the mask and the image as a tuple"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    if percentage is None:
        perc = int((np.sum(center) / 2) * .70)
    else:
        perc = int((np.sum(center) / 2) * percentage)

    mask = np.zeros(image.shape[:2], dtype='uint8')
    cv.rectangle(mask, np.subtract(center, perc), np.add(center, perc), 255, -1)

    return (mask, cv.bitwise_and(image, image, mask=mask))

def plotHistogram(img, title, mask=None, grayScale=None):
    """It plot the histogram of an image. This method is able to plot histograrm of grayscal as well as 3 channel color image."""
    plt.figure()
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('No of pixels')

    if grayScale is None:
        channels= cv.split(img)
        colors = ('b', 'g', 'r')

        for (channel, color) in zip(channels, colors):
            hist = cv.calcHist([channel], [0], mask, [256], [0, 256])
            plt.xlim([0, 256])
            plt.plot(hist)

    if grayScale is True:
        hist = cv.calcHist([img], [0], mask, [256], [0, 256])
        plt.xlim([0, 256])
        plt.plot(hist)
        plt.show()