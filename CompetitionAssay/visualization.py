"""
Author: Eric Schmidt

This file contains all classes that are connected to visualiztion
"""

import matplotlib.pyplot as plt
import numpy as np
from CompetitionAssay.datahandling import GetNormalizationFactor


def PlotCompetitionsHistogram(WT_single_cell_area, Mutant_single_cell_area, visualize=False):
    """
    This function plots histograms of the single cell areas of WT and Mutant for a single
    competition (single image).
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    max_value = np.max([WT_single_cell_area.max(), Mutant_single_cell_area.max()])
    bins = np.linspace(0, max_value, 100)

    hist1, bins1 = np.histogram(WT_single_cell_area, bins=bins)
    hist2, bins2 = np.histogram(Mutant_single_cell_area, bins=bins)

    hist1_log = np.log10(hist1)
    hist2_log = np.log10(hist2)

    # normalize histogram
    hist1 = hist1_log / np.nansum(hist1_log[hist1_log > 0])
    hist2 = hist2_log / np.nansum(hist2_log[hist2_log > 0])

    plt.bar(bins1[:-1], hist1 + 1, width=np.diff(bins1), align="edge", alpha=0.5, label="WT")
    plt.bar(bins2[:-1], hist2 + 1, width=np.diff(bins2), align="edge", alpha=0.5, label="Mutant")

    plt.plot([0, max_value], [1, 1], "k-.", label="1 line")

    plt.ylim([0.9, np.max([hist1, hist2]) + 1.1])
    plt.legend()
    plt.xlabel("Area[px]")
    plt.ylabel("log(Count) + 1, Normalized")

    if visualize:
        plt.show()
    return fig


def Plot_Area_Histogram_Overall(
    all_single_areas_WT, all_single_areas_Mutant, competition, output_dir, visualize=False
):
    """
    This function plot the area distribution of all mutants and WTs in all images in the folder.
    So every image of a competition gives N_i values for mutant and M_i values for WT
    and the histogram is plotted over all N_i and M_i values.
    """
    max_value = np.max([all_single_areas_WT.max(), all_single_areas_Mutant.max()])
    bins = np.linspace(0, max_value, 100)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    hist1, bins1 = np.histogram(all_single_areas_WT, bins=bins)
    hist2, bins2 = np.histogram(all_single_areas_Mutant, bins=bins)
    hist1_log = np.log10(hist1)
    hist2_log = np.log10(hist2)

    plt.bar(bins1[:-1], hist1_log + 1, width=np.diff(bins1), align="edge", alpha=0.5, label="WT")
    plt.bar(
        bins2[:-1], hist2_log + 1, width=np.diff(bins2), align="edge", alpha=0.5, label="Mutant"
    )

    plt.plot([0, np.max(bins2)], [1, 1], color="black", linestyle="--", linewidth=2, label="1")

    plt.legend()
    plt.xlabel("Area[px]")
    plt.ylabel("log10(Count) + 1")
    plt.title("Histogram of single cell areas")

    if visualize:
        plt.show()

    else:
        plt.close()

    return fig


def OverlaySegmentationMulticlass(img1, segmentation1, img2, segmentation2):
    """This function can be used to plot the overlay of the segmentation contour on the image when
    having 3 classes output (e.g. directly after making the RFC classification, before binarizing
    the image). The function is not used in the pipeline, because I directly force the RFC output
    to be binary, but the function can still be used for visualization."""

    # Create a figure and axis with 2 plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img1, cmap="gray", vmin=np.min(img1), vmax=2 * np.mean(img1))

    # Create a mask for the contour
    contour_mask_1 = np.zeros_like(segmentation1)
    contour_mask_2 = np.zeros_like(segmentation1)
    contour_mask_1[segmentation1 == 1] = 1  # Change the value based on your label
    contour_mask_2[segmentation1 == 2] = 2  # Change the value based on your label

    # Plot segmentation contour
    contour_2 = ax[0].contour(contour_mask_2, colors="red", linewidths=0.4)
    contour_1 = ax[0].contour(contour_mask_1, colors="cyan", linewidths=0.5)

    # Customize the contour appearance if needed
    # For example, you can set the transparency of the contour:
    contour_1.set_alpha(0.5)
    contour_2.set_alpha(0.5)

    ax[1].imshow(img2, cmap="gray", vmin=np.min(img2), vmax=2 * np.mean(img2))

    # Create a mask for the contour
    contour_mask_1 = np.zeros_like(segmentation2)
    contour_mask_2 = np.zeros_like(segmentation2)
    contour_mask_1[segmentation2 == 1] = 1  # Change the value based on your label
    contour_mask_2[segmentation2 == 2] = 2  # Change the value based on your label

    # Plot segmentation contour
    contour_2 = ax[1].contour(contour_mask_2, colors="red", linewidths=0.4)
    contour_1 = ax[1].contour(contour_mask_1, colors="cyan", linewidths=0.5)

    # Customize the contour appearance if needed
    # For example, you can set the transparency of the contour:
    contour_1.set_alpha(0.5)
    contour_2.set_alpha(0.5)

    # set title
    ax[0].set_title("WT")
    ax[1].set_title("Mutant")

    return ax, fig


def OverlaySegmentation(img1, segmentation1, img2, segmentation2):
    """This function can be used to plot the overlay of the segmentation contour on the image when
    having 2 classes output"""

    # Create a figure and axis with 2 plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].imshow(img1, cmap="gray", vmin=np.min(img1), vmax=2 * np.mean(img1))

    # Create a mask for the contour
    contour_mask_1 = np.zeros_like(segmentation1)
    contour_mask_1[segmentation1 == 1] = 1  # Change the value based on your label

    # Plot segmentation contour
    contour_1 = ax[0].contour(contour_mask_1, colors="cyan", linewidths=0.5)

    # Customize the contour appearance if needed
    # For example, you can set the transparency of the contour:
    contour_1.set_alpha(0.5)

    ax[1].imshow(img2, cmap="gray", vmin=np.min(img2), vmax=2 * np.mean(img2))

    # Create a mask for the contour
    contour_mask_1 = np.zeros_like(segmentation2)
    contour_mask_1[segmentation2 == 1] = 1  # Change the value based on your label

    # Plot segmentation contour
    contour_1 = ax[1].contour(contour_mask_1, colors="cyan", linewidths=0.5)

    # Customize the contour appearance if needed
    # For example, you can set the transparency of the contour:
    contour_1.set_alpha(0.5)

    # set title
    ax[0].set_title("WT")
    ax[1].set_title("Mutant")

    return ax, fig


def Plot_Biofilm_Identification(WT_intensity, Mutant_intensity, WT_area, Mutant_area):
    max_WT_approx = GetNormalizationFactor(WT_intensity)
    max_Mutant_approx = GetNormalizationFactor(Mutant_intensity)

    # plot intensity vs area for both types
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(WT_area, WT_intensity / max_WT_approx, label="WT", color="b", alpha=0.5)
    ax.scatter(
        Mutant_area,
        Mutant_intensity / max_Mutant_approx,
        label="Mutant",
        color="k",
        alpha=0.5,
    )

    # get maximum of the areas
    max_area = max(*WT_area, *Mutant_area)
    max_intensity = max(*WT_intensity / max_WT_approx, *Mutant_intensity / max_Mutant_approx)
    min_intensity = min(*WT_intensity / max_WT_approx, *Mutant_intensity / max_Mutant_approx)

    plt.hlines(y=1, xmin=0, xmax=max_area, color="k", linestyle="--", linewidth=2)
    plt.vlines(
        x=800, ymin=min_intensity, ymax=max_intensity, color="k", linestyle="--", linewidth=2
    )

    # add text
    plt.text(0.6 * max_area, max_intensity, "Large Biofilms", fontsize=15)
    plt.text(0.6 * max_area, min_intensity, "Large Dead Biofilms", fontsize=15)

    plt.text(0, max_intensity, "Small Agglomerations", fontsize=15)
    plt.text(0, min_intensity, "Small single events", fontsize=15)

    plt.xlabel("Area[px]")
    plt.ylabel("Intensity")
    plt.legend()

    return ax, fig
