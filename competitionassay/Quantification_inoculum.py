"""
author: Eric Schmidt

This script is used to quantify the data.
It needs binary images as input and produces
... as output
"""
import os
from skimage.io import imread, imsave
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
from skimage.measure import label, regionprops_table
import nd2
from skimage.morphology import binary_erosion, binary_dilation, remove_small_objects


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


def VisualizeSegmentation(intensity_img, binary_img, df):
    # visualize the image and scatter plot of area vs mean intensity
    fig = plt.figure(figsize=(15, 5))
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(intensity_img)
    ax1.set_title("intensity_img")

    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax2.imshow(binary_img)
    ax2.set_title("binary_img")

    ax3 = plt.subplot(1, 3, 3)
    ax3.scatter(df["area"], df["mean_intensity"])
    ax3.set_title("area vs mean_intensity")
    ax3.set_xlabel("area")
    ax3.set_ylabel("mean_intensity")

    plt.savefig("Segmentation/mutant1.png", dpi=500)
    plt.show()


def main():
    binary_fn = "predictions_transwell.tif"
    intensity_fn = "denoised/WT_C2-MAX_20230424_5hpif_mix2_WTmScarlet_dwspFmNeonGreen_ours_R2_001-1.tif"

    binary_img = imread(binary_fn)
    intensity_img = imread(intensity_fn)

    # split clusters
    binary_img = ChangeMorphology(binary_img, 4, show=False)

    # get labels with skimage
    labels = label(binary_img, connectivity=1)

    # drop small clusters
    labels = remove_small_objects(
        labels, min_size=16
    )  # arbitrarily (maybe choose based on histogram)

    # get regionprops with skimage (area and intensity per label) from intensity and label image
    props = regionprops_table(
        labels,
        intensity_img,
        properties=(
            "label",
            "area",
            "mean_intensity",
            "max_intensity",
            "min_intensity",
        ),
    )

    # get area and intensity per label with polars
    df = pl.DataFrame(props)

    VisualizeSegmentation(intensity_img, binary_img, df)

    # plot histogram of area
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hist(df["area"], bins=100)
    axs[0].set_title("pixels of the labels")
    axs[0].set_xlabel("pixels")
    axs[0].set_ylabel("count")

    # plot scatterplot of area vs mean intensity
    axs[1].scatter(df["area"], df["mean_intensity"])
    axs[1].set_title("area vs mean_intensity")
    axs[1].set_xlabel("area")
    axs[1].set_ylabel("mean_intensity")

    plt.show()

    # get the toal sum of area of the image
    total_area = df["area"].sum()

    print(
        "Total amount of pixel: %i, this are %.3f percent of the image"
        % (
            total_area,
            100 * total_area / (intensity_img.shape[0] * intensity_img.shape[1]),
        )
    )

    print("done")


if __name__ == "__main__":
    print(__doc__)
    main()
