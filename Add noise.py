############################################################
# Magikcode - Getting started with images and opencv.
# May 12, 2017
############################################################

import cv2
import numpy as np

def show_image_and_wait(title, image):
    # Display the image in a window.  Window size fits image.
    cv2.imshow(title, image)

    # Wait for user input; click X to destroy window.
    cv2.waitKey(0)

    # Destroy window and return to caller.
    cv2.destroyAllWindows()

def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))

    h = temp_image.shape[0]
    w = temp_image.shape[1]
    noise = np.random.randn(h, w) * noise_sigma

    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:
        noisy_image = temp_image + noise
    else:
        noisy_image[:,:,0] = temp_image[:,:,0] + noise
        noisy_image[:,:,1] = temp_image[:,:,1] + noise
        noisy_image[:,:,2] = temp_image[:,:,2] + noise

    """
    print('min,max = ', np.min(noisy_image), np.max(noisy_image))
    print('type = ', type(noisy_image[0][0][0]))
    """

    return noisy_image

def convert_to_uint8(image_in):
    temp_image = np.float64(np.copy(image_in))
    cv2.normalize(temp_image, temp_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)

    return temp_image.astype(np.uint8)

def main():
    girl_face_filename = "girl_face_closeup.jpg"
    print('opening image: ', girl_face_filename)

    # cv2.IMREAD_COLOR - read in color images (BGR)
    # cv2.IMREAD_GRAYSCALE - convert image to grayscale
    girl_face_image = cv2.imread(girl_face_filename, cv2.IMREAD_UNCHANGED)
    girl_face_grayscale_image = cv2.cvtColor(girl_face_image, cv2.COLOR_BGR2GRAY)

    """
    Gaussian is a nice noise function.  Gaussian noise are values generated from the
    random normal distribution.  The mean of the distribution is 0 and the standard
    deviation is 1.  The standard deviation is a measure of how spread out the values
    are from the mean or 0.  randn() generates random numbers from this distribution.
    The Gaussian distribution is symmetric about the mean of the probability.

    Sigma determines the magnitude of the noise function.  For a small sigma, the noise
    function produces values very close to zero or a gray image since we want to map the
    pixel with a value of zero to gray.  The larger sigma spreads out the noise.
    Multiplying an image by a noise image generated from a Gaussian function effectively
    changes the standard deviation of the pixel values.  This is how far apart the pixel
    colors are in value.
    """
    noisy_sigma = 35
    noisy_image = add_gaussian_noise(girl_face_grayscale_image, noisy_sigma)

    print('noisy image shape: {0}, len of shape {1}'.format(\
        girl_face_image.shape, len(noisy_image.shape)))
    print('    WxH: {0}x{1}'.format(noisy_image.shape[1], noisy_image.shape[0]))
    print('    image size: {0} bytes'.format(noisy_image.size))

    show_image_and_wait(girl_face_filename, convert_to_uint8(noisy_image))
    noisy_filename = 'girl_face_noise_' + str(noisy_sigma) + '.jpg'
    cv2.imwrite(noisy_filename, noisy_image)

if __name__ == "__main__":
    main()
