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

    plt.figure()
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
    plt.savefig(output_dir + "area_distribution_" + competition[:-1] + ".png", dpi=500)
    plt.savefig(output_dir + "area_distribution_" + competition[:-1] + ".pdf", dpi=500)

    if visualize:
        plt.show()

    else:
        plt.close()

    print("Histogram saved to: " + output_dir + "area_distribution_" + competition[:-1] + ".png")


def OverlaySegmentation(img1, segmentation1, img2, segmentation2):
    import matplotlib.pyplot as plt
    import numpy as np

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




def Plot_Biofilm_Identification(
    WT_intensity, Mutant_intensity, WT_area, Mutant_area, WT_file, output_dir
):
    max_WT_approx = GetNormalizationFactor(WT_intensity)
    max_Mutant_approx = GetNormalizationFactor(Mutant_intensity)

    # plot intensity vs area for both types
    plt.figure()
    plt.scatter(WT_area, WT_intensity / max_WT_approx, label="WT", color="b", alpha=0.5)
    plt.scatter(
        Mutant_area,
        Mutant_intensity / max_Mutant_approx,
        label="Mutant",
        color="k",
        alpha=0.5,
    )
    plt.xlabel("Area[px]")
    plt.ylabel("Intensity")
    plt.legend()
    plt.savefig(output_dir + WT_file[:-4] + "_intensity_vs_area.png", dpi=500)

