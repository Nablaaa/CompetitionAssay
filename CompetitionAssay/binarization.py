"""
Author: Eric Schmidt

This file contains all classes that are connected to binarization,which is RFC and post processing.
"""

from skimage.morphology import binary_erosion, binary_dilation, remove_small_objects
import CompetitionAssay import datahandling
import numpy as np
import pickle
import matplotlib.pyplot as plt


def RemoveSmallObjects(segmentation, min_size=9):
    """
    Remove small objects from a segmentation
    """
    # segmentation can have multiple labels, so work with binary
    binary = np.zeros_like(segmentation)
    binary[segmentation > 0] = 1
    segmentation_without_small_objects = remove_small_objects(
        binary.astype("bool"), min_size=min_size
    )

    # convert back to original labels
    segmentation[segmentation_without_small_objects == 0] = 0
    return segmentation


# # pick out a certain label (e.g. when classifier has 2 categories)
# def PickLabel(img, label):
#     img[img != label] = 0
#     img[img == label] = 1
#     return img


class myRFC:
    """Initialize the RFC, load the model and wait for data
    Input: model

    Functions:
    predict(img)
        returns binary image
    visualize()
        shows the input and the output
    """

    def __init__(self, modelpath: str) -> None:
        self.classifier = pickle.load(open(modelpath, "rb"))

    def predict(self, img: np.ndarray) -> np.ndarray:
        """binarizes the image"""

        self.img = img
        img_shape = img.shape
        feature_stack = datahandling.generate_feature_stack(self.img)

        result_1d = self.classifier.predict(feature_stack.T) - 1
        self.result_2d = result_1d.reshape(img_shape)
        self.result_2d[self.result_2d != 1] = 0

        return self.result_2d

    def visualize(self) -> None:
        """Visualize the segmentation"""
        # create a plot with 2 subplots and add the img and the result_2d
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(self.img)
        axs[0].set_title("input")
        axs[1].imshow(self.result_2d)
        axs[1].set_title("output")
        plt.show()


def ChangeMorphology(binary_img, N, show=False):
    for i in range(N):
        binary_img = binary_erosion(binary_img, footprint=np.ones((3, 3)))
        binary_img = binary_dilation(binary_img, footprint=np.ones((3, 3)))

    if show:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))

        axs[0, 0].imshow(binary_img)
        axs[0, 0].set_title("binary_img")

        binary_img = ChangeMorphology(binary_img, 1)
        axs[0, 1].imshow(binary_img)
        axs[0, 1].set_title("1")

        binary_img = ChangeMorphology(binary_img, 2)
        axs[1, 0].imshow(binary_img)
        axs[1, 0].set_title("2")

        binary_img = ChangeMorphology(binary_img, 3)
        axs[1, 1].imshow(binary_img)
        axs[1, 1].set_title("3")

        plt.show()

    return binary_img
