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
base_dir = "dataset_competition_assays/competition_2_WTmScarlet_dwspFmNeonGreen/TW_growth/"

# output directory
output_dir = base_dir + 'binary/'

# create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    

modelpath = "models/RandomForestClassifier_transwell/random_forest_classifier_transwell_denoised.pkl"


# get all the files in the directory
filenames = os.listdir(base_dir + 'denoised/')

for fn in filenames:
    print(fn)
    img_denoised = imread(base_dir + 'denoised/' + fn)
    binary_img = helpers.RandomForestSegmentation(img_denoised, modelpath, visualize=False)

    imsave(output_dir + 'binary_' + fn, binary_img)    



print('done')