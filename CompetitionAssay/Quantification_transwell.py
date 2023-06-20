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
from skimage.morphology import remove_small_objects
import helpers


def main():
    # define base directory
    base_dir = "dataset_competition_assays/"
    files_are_in = "/TW_growth/"

    # get all the files in the directory
    different_competitions_folders = os.listdir(base_dir)

    for competition in different_competitions_folders[1:]:
        WT_files, Mutant_files = helpers.GetTranswellData(
            base_dir, competition, files_are_in
        )

        for WT_file, Mutant_file in zip(WT_files, Mutant_files):
            print(WT_file, Mutant_file)

            WT_img = imread(base_dir + competition + files_are_in + WT_file)
            Mutant_img = imread(base_dir + competition + files_are_in + Mutant_file)


            # start here the data normalization
            WT_normalized = helpers.NormalizeImg(WT_img)
            Mutant_normalized = helpers.NormalizeImg(Mutant_img)

            # start here the denoising with a pretrained model
            # see https://colab.research.google.com/drive/18PiNcg6t73GwjJqFbaYiVLJzSjYzFn2F
            WT_denoised = helpers.n2vDenoising(WT_normalized, visualize=False, pass_it=True)
            Mutant_denoisded = helpers.n2vDenoising(Mutant_normalized, visualize=True,pass_it=True)

           
            
            # start here the segmentation with a pretrained model
            modelpath = "models/RandomForestClassifier_transwell/random_forest_classifier_transwell_denoised.pkl"
            WT_segmented = helpers.RandomForestSegmentation(WT_denoised, modelpath, visualize=True)
            Mutant_segmented = helpers.RandomForestSegmentation(Mutant_denoisded, modelpath, visualize=True)


            # TODO: make a github repository for this

            # start here the quantification A (competition based on area)

            # start here the quantification B (count biofilms)

            # output is a csv file that contains for each competition the 2 values

        # output is a csv file that contains for each folder the 5 values

    # output is a csv file that contains the outputs of all 7 folders

    print("next")

    binary_fn = "predictions_transwell.tif"
    intensity_fn = "denoised/WT_C2-MAX_20230424_5hpif_mix2_WTmScarlet_dwspFmNeonGreen_ours_R2_001-1.tif"

    binary_img = imread(binary_fn)
    intensity_img = imread(intensity_fn)

    # split clusters
    binary_img = helpers.ChangeMorphology(binary_img, 4, show=False)

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

    helpers.VisualizeSegmentation(intensity_img, binary_img, df)

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
