"""
Author: Eric Schmidt

This file contains all classes that are connected to quantification
"""


import numpy as np
from CompetitionAssay.binarization import ChangeMorphology
from CompetitionAssay.datahandling import GetNormalizationFactor
from skimage.measure import label, regionprops
from typing import Tuple
from matplotlib import pyplot as plt


def GetCoveredAreaPercent(binary_img: np.ndarray) -> float:
    assert np.unique(binary_img).size == 2, "binary_img must be binary"
    return np.sum(binary_img) / (binary_img.shape[0] * binary_img.shape[1])


def GetSingleCellAreaAndIntensity(
    binary_img: np.ndarray, intensity_img: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the area of a single cell in a binary image"""
    assert np.unique(binary_img).size == 2, "binary_img must be binary"
    assert (
        binary_img.shape == intensity_img.shape
    ), "binary_img and intensity_img must have the same shape"

    ChangeMorphology(binary_img, 2, show=False)

    # label the image
    labeled_img = label(binary_img)

    # get the area and intensityof each object
    props = regionprops(labeled_img, intensity_image=intensity_img)

    # get all areas
    areas = [prop.area for prop in props]

    # get all mean intensities
    mean_intensities = [prop.mean_intensity for prop in props]

    return np.array(areas), np.array(mean_intensities)


def LocalCompetition(
    img_x: np.ndarray,
    binary_x: np.ndarray,
    img_y: np.ndarray,
    binary_y: np.ndarray,
    visualize: bool = False,
) -> Tuple[float, float, float, float]:
    """
    This function takes two images and their segmentation and calculates the local competition.

    So it uses segmentation_x to define a mask of the area of interest,
    then it calculates the overlap with mask_y to get the overlapping area
    (this measures how much competition has X in the location in which it lives)

    Then it measures the intensity of all clusters of Y in the area of overlap with X,
    so that an intensity density can be calculated
    (this is a symmetric output, so X and Y can be swapped in the function but the result stays
    the same)

    """

    # get the mask of the area of interest
    mask_x = binary_x > 0
    mask_y = binary_y > 0
    overlap = mask_x * mask_y

    # total area of the mask
    total_area_x = np.sum(mask_x)
    total_area_y = np.sum(mask_y)

    # area competition of x in y
    area_competition_x_in_y = np.sum(overlap) / total_area_y
    area_competition_y_in_x = np.sum(overlap) / total_area_x

    # get the overlapping area
    overlapping_area = np.sum(overlap)

    print(
        "X competes against Y in %.2f percent of the area that is colonized by Y, so in %.2f percent of the area that is colonized by Y, Y has no competitor"
        % (area_competition_x_in_y * 100, (1 - area_competition_x_in_y) * 100)
    )
    print(
        "Y competes against X in %.2f percent of the area that is colonized by X, so in %.2f percent of the area that is colonized by X, X has no competitor"
        % (area_competition_y_in_x * 100, (1 - area_competition_y_in_x) * 100)
    )

    # now consider the competition in the overlap region to find out who has more dominance there

    # normalize intensity of the image (based on something like the maximum intensity)
    norm_factor_x = GetNormalizationFactor(img_x)
    img_x = img_x / norm_factor_x

    norm_factor_y = GetNormalizationFactor(img_y)
    img_y = img_y / norm_factor_y

    # get the intensity of all clusters of x and y in the overlap region
    x_in_overlap = img_x[overlap]
    y_in_overlap = img_y[overlap]

    # get the intensity density in the overlap region
    normalized_intensity_density_x = np.sum(x_in_overlap) / overlapping_area
    normalized_intensity_density_y = np.sum(y_in_overlap) / overlapping_area

    # get the intensity density in the region where there is mask_x but not mask_y
    x_alone = mask_x * np.logical_not(mask_y)
    normalized_intensity_density_x_alone = np.sum(img_x[x_alone]) / np.sum(x_alone)

    y_alone = mask_y * np.logical_not(mask_x)
    normalized_intensity_density_y_alone = np.sum(img_y[y_alone]) / np.sum(y_alone)

    # get the intensity density in the overall region
    normalized_intensity_density_x_total = np.sum(img_x[mask_x]) / total_area_x
    normalized_intensity_density_y_total = np.sum(img_y[mask_y]) / total_area_y

    print(
        "X has an intensity density of %.2f in the overlap region, Y has an intensity density of %.2f in the overlap region. For comparison: X has an intensity density of %.2f in the total region, Y has an intensity density of %.2f in the total region"
        % (
            normalized_intensity_density_x,
            normalized_intensity_density_y,
            normalized_intensity_density_x_total,
            normalized_intensity_density_y_total,
        )
    )

    if visualize == True:
        # visualize in the image the different regions with different colors (overlap region, distinct x/y regions)
        plt.figure()

        distinct_x = np.logical_and(mask_x, np.logical_not(overlap))
        distinct_y = np.logical_and(mask_y, np.logical_not(overlap))

        final_img = np.zeros((img_x.shape[0], img_x.shape[1]))
        final_img[distinct_x] = 1
        final_img[distinct_y] = 2
        final_img[overlap] = 3

        plt.imshow(final_img)

        plt.figure()
        plt.imshow(img_x)
        plt.title("img x")

        plt.figure()
        plt.imshow(img_y)
        plt.title("img y")

        plt.show()

    return (
        area_competition_x_in_y,
        area_competition_y_in_x,
        normalized_intensity_density_x,
        normalized_intensity_density_y,
        normalized_intensity_density_x_alone,
        normalized_intensity_density_y_alone,
        normalized_intensity_density_x_total,
        normalized_intensity_density_y_total,
    )
