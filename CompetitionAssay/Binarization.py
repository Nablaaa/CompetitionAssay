"""
Author: Eric Schmidt



This script gives an example of how to set up the RFC in a workflow that loads WT and mutant at the same time.

The input is the model path, as well as the folder that consists the competition assay files.

Later it will be integrated in a workflow (Quantification_transwell.py)
"""

import os
from skimage.io import imread, imsave
import helpers

# define base directory
base_dir = "dataset_competition_assays/"
competition = "competition_2_WTmScarlet_dwspFmNeonGreen/"
files_are_in = "TW_growth/"

modelpath_WT = "models/RandomForestClassifier_transwell/transwell_denoised_2_categories_WT.pkl"
modelpath_Mutant = (
    "models/RandomForestClassifier_transwell/transwell_denoised_2_categories_Mutant.pkl"
)

# objects have to be at least 49 objects (e.g. 7x7) large
min_size = 49  # pixels

# output directory
output_dir = base_dir + competition + files_are_in + "binary/"

# create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# get all filenames
WT_denoised_files, Mutant_denoised_files = helpers.GetTranswellData(
    base_dir, competition, files_are_in + "denoised/"
)

for WT_denoised_file, Mutant_denoised_file in zip(WT_denoised_files, Mutant_denoised_files):
    print(WT_denoised_file)
    WT_denoised = imread(base_dir + competition + files_are_in + "denoised/" + WT_denoised_file)
    # TODO: use the helpers.n2vDenoising function

    WT_RFC = helpers.myRFC(modelpath_WT)
    WT_binary = WT_RFC.predict(WT_denoised)
    WT_without_small_objects = helpers.RemoveSmallObjects(WT_binary, min_size=min_size)

    imsave(output_dir + WT_denoised_file, WT_without_small_objects)

    print(Mutant_denoised_file)
    Mutant_denoised = imread(
        base_dir + competition + files_are_in + "denoised/" + Mutant_denoised_file
    )

    mutant_RFC = helpers.myRFC(modelpath_Mutant)
    Mutant_binary = mutant_RFC.predict(Mutant_denoised)
    Mutant_without_small_objects = helpers.RemoveSmallObjects(Mutant_binary, min_size=min_size)

    imsave(output_dir + Mutant_denoised_file, Mutant_without_small_objects)

print("done")
