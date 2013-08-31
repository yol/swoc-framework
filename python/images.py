#!/usr/bin/python

import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data
from skimage.viewer import CollectionViewer

CLASSES = ["rabbit", "airplane", "bicycle"]

def opencv():
    import cv2
    im = cv2.imread("../images_all/rabbit/13333.png")

    plt.imshow(im)


def generate_classes(classes):
    root = "../images_all"
    for cls in classes:
        clspath = os.path.join(root, cls)
        print clspath
        for rt, dirs, files in os.walk(clspath):
            for fil in files:
                yield (fil, cls)

def generate_patch_rois(imsize, stride=50, patch_size=32):
    xsize, ysize = imsize

    for x in range(0, xsize-patch_size+1, stride):
        for y in range(0, ysize-patch_size+1, stride):
            yield ((x,x+patch_size), (y,y+patch_size))

def generate_patches(image):
    for roi in generate_patch_rois(image.shape, image.shape[0]/28):
        print roi
        yield image[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]


def skimage(image, vis=False):
    #From http://scikit-image.org/docs/dev/auto_examples/plot_hog.html
    #image = data.load("/home/loy/Development/swoc-framework/images_all/rabbit/13333.png")

    if vis:
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True, normalise=False)
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
    else:
        fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                cells_per_block=(1, 1), visualise=False, normalise=False)

    return fd

if __name__ == "__main__":
    image = data.load("/home/loy/Development/swoc-framework/images_all/rabbit/13333.png")
    #print skimage(image, False)

    plt.imshow(image)
    plt.show()

    print len(list(generate_classes(CLASSES)))

    #import ipdb; ipdb.set_trace()
    p = generate_patches(image)
    patches = list(p)
    print len(patches)

    plt.imshow(patches[0])
    plt.show()