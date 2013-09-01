#!/usr/bin/python

import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from skimage.viewer import CollectionViewer
from skimage.io import ImageCollection
import numpy as np
from sklearn.cluster import MiniBatchKMeans

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

def generate_patch_rois(imsize, stride, patch_size=32):
    xsize, ysize = imsize

    for x in range(0, xsize-patch_size+1, stride):
        for y in range(0, ysize-patch_size+1, stride):
            yield ((x,x+patch_size), (y,y+patch_size))

def generate_patches(image):
    for roi in generate_patch_rois(image.shape, image.shape[0]/28):
        #print roi
        yield image[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]


def skimage(image, vis=False):
    #From http://scikit-image.org/docs/dev/auto_examples/plot_hog.html
    #image = data.load("/home/loy/Development/swoc-framework/images_all/rabbit/13333.png")

    parameters = {"orientations":4, "pixels_per_cell":(8, 8),
                "cells_per_block":(4, 4), "normalise":False}

    if vis:
        fd, hog_image = hog(image, visualise=True, **parameters)
        plt.figure(figsize=(8, 4))

        plt.subplot(121).set_axis_off()
        plt.imshow(image, cmap=plt.cm.gray)
        plt.title('Input image')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

        # plt.subplot(122).set_axis_off()
        # plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        # plt.title('Histogram of Oriented Gradients')


        plt.subplot(122).set_axis_off()
        plt.plot(fd)
        plt.title('HoG')
        plt.show()
    else:
        fd = hog(image, visualise=False, **parameters)

    return fd

def generate_features(image):
    """Given an input image (with size 256 x 256 pixels), split the image into 28 x 28 blocks. 
    Each block is 32 x 32 pixels (the blocks will overlap)"""

    resized = image.copy()
    resized.resize((256, 256))

    patches = generate_patches(image)

    HoGs = map(skimage, patches)
    return HoGs

def get_collection_descriptors(clas):
    """Returns an array of 80 images, with 784 patch descriptors, each of which is a 64-length vector"""
    filename = "{0}.npy".format(clas)
    if not os.path.exists(filename):
        coll = ImageCollection("/home/loy/Development/swoc-framework/images_all/{0}/*.png".format(clas))

        class_descriptors = map(generate_features, coll)
        desc = np.array([list(cls_desc) for cls_desc in class_descriptors])

        np.save(filename, desc)

        return desc
    else:
        return np.load(filename)

def get_patch_descriptors():
    patch_descriptors_filename = "patch_descriptors.npy"
    if not os.path.exists(patch_descriptors_filename):
        cls_descriptors = {cls:get_collection_descriptors(cls) for cls in CLASSES}

        all_descriptors = [np.reshape(cls_descriptors[clas], (80*784, 64)) for clas in CLASSES]
        patch_descriptors = np.concatenate(all_descriptors)

        np.save(patch_descriptors_filename, patch_descriptors)
    else:
        patch_descriptors = np.load(patch_descriptors_filename)

    return patch_descriptors

if __name__ == "__main__":
    image = data.load("/home/loy/Development/swoc-framework/images_all/rabbit/13333.png")
    #print skimage(image, False)

    #print len(list(generate_classes(CLASSES)))

    #import ipdb; ipdb.set_trace()
    #p = generate_features(image)
    # patch_hogs = list(p)
    # print patch_hogs
    # print len(patch_hogs)

    patch_descriptors = get_patch_descriptors()