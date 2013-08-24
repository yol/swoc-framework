#!/usr/bin/python

def opencv():
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    im = cv2.imread("../images_all/rabbit/13333.png")

    plt.imshow(im)

def skimage():
    import matplotlib.pyplot as plt

    from skimage.feature import hog
    from skimage import data, color, exposure

    image = data.load("/home/loy/Development/swoc-framework/images_all/rabbit/13333.png")

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)

    plt.figure(figsize=(8, 4))

    plt.subplot(121).set_axis_off()
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

    plt.subplot(122).set_axis_off()
    plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    plt.title('Histogram of Oriented Gradients')
    plt.show()
