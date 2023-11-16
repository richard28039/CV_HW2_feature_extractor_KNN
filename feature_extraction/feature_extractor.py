import cv2
import numpy as np
from skimage import feature

def Gabor_filter(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ksize = 7
    sigma = 1
    theta = 0
    lamda = 4
    gamma = 0.5
    phi = 0     

    gabor_filter = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi)

    filtered_image = cv2.filter2D(image, cv2.CV_8UC3, gabor_filter)

    return filtered_image

def lbp(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    radius = 1
    n_points = 8 * radius

    lbp = feature.local_binary_pattern(image, P=n_points, R=radius, method='uniform')
    lbp_contrast = (lbp - lbp.min()) / (lbp.max() - lbp.min()) * 255
    return lbp_contrast

def RGB_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
    return hist

# if __name__ == "__main__":