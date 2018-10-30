import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy import misc

# Ensure plots embeded in notebook
# %matplotlib inline

def noise_generator(noise_type, image):
    """
    Generate noise to a given Image based on required noise type
    Input parameters:
        image: ndarray (input image data. It will be converted to float)

        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row, col, ch = image.shape
    if noise_type == "gauss":
        mean = 0
        # var = 1.1
        sigma = 50
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy.astype('uint8')
        # return gauss.astype('uint8')
    elif noise_type == "®s&p":
        s_vs_p = 0.5
        amount = 0.4
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 1 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy.astype('uint8')
    elif noise_type == "speckle":
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy.astype('uint8')
    else:
        return image


if __name__ == '__main__':
    fn = "timg.jpeg"
    img = cv2.imread(fn)
    img2 = noise_generator("gauss", img)
    cv2.namedWindow('img')
    cv2.imshow('img', img2)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # fn = "timg.jpeg"
    # img = cv2.imread(fn)
    # noise_generator("s&p", img)
    # cv2.namedWindow('img')
    # cv2.imshow('img', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # fn = "timg.jpeg"
    # img = cv2.imread(fn)
    # img2 = noise_generator("poisson", img)
    # cv2.namedWindow('img')
    # cv2.imshow('img', img2)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
