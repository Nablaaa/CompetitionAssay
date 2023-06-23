"""
Author: Eric Schmidt

This script is a temporarily solution to just load images and make predictions
with the random forest classifier

Later it will be integrated in a workflow (Quantification_transwell.py)
"""

import os
from skimage.io import imread, imsave
import numpy as np
import helpers
import matplotlib.pyplot as plt

# define base directory
base_dir = "dataset_competition_assays/"
competition = "competition_2_WTmScarlet_dwspFmNeonGreen/"
files_are_in = "TW_growth/"

modelpath_WT = "models/RandomForestClassifier_transwell/transwell_denoised_2_categories_WT.pkl"
modelpath_Mutant = (
    "models/RandomForestClassifier_transwell/transwell_denoised_2_categories_Mutant.pkl"
)

# objects have to be at least 49 objects (e.g. 7x7) large
min_size = 49

# output directory
output_dir = base_dir + competition + files_are_in + "segmentation/"

# create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# get all filenames
WT_denoised_files, Mutant_denoised_files = helpers.GetTranswellData(
    base_dir, competition, files_are_in + "binary/"
)

for WT_denoised_file, Mutant_denoised_file in zip(WT_denoised_files, Mutant_denoised_files):
    print(WT_denoised_file)
    WT_denoised = imread(base_dir + competition + files_are_in + "denoised/" + WT_denoised_file)
    WT_segmented = helpers.RandomForestSegmentation(WT_denoised, modelpath_WT, visualize=False)
    WT_without_small_objects = helpers.RemoveSmallObjects(WT_segmented, min_size=min_size)

    imsave(output_dir + WT_denoised_file, WT_without_small_objects)

    Mutant_denoised = imread(
        base_dir + competition + files_are_in + "denoised/" + Mutant_denoised_file
    )
    Mutant_segmented = helpers.RandomForestSegmentation(
        Mutant_denoised, modelpath_Mutant, visualize=False
    )
    Mutant_without_small_objects = helpers.RemoveSmallObjects(Mutant_segmented, min_size=min_size)
    imsave(output_dir + Mutant_denoised_file, Mutant_without_small_objects)

print("done")
