"""
Author: Eric Schmidt

This file contains all classes that help to load, save
and in general handle the data.
"""

import numpy as np
from skimage import filters
import os
from typing import List


def NormalizeImg(img):
    """Normalize img input by subtracting the mean
    and dividing by std"""
    img = img.astype("float32")
    img -= np.mean(img)
    img /= np.std(img)

    return img


def generate_feature_stack(image):
    # determine features
    blurred = filters.gaussian(image, sigma=2)
    edges = filters.sobel(blurred)

    # collect features in a stack
    # The ravel() function turns a nD image into a 1-D image.
    # We need to use it because scikit-learn expects values
    # in a 1-D format here.
    feature_stack = [image.ravel(), blurred.ravel(), edges.ravel()]

    # return stack as numpy-array
    return np.asarray(feature_stack)


def format_data(feature_stack, annotation):
    # reformat the data to match what scikit-learn expects
    # transpose the feature stack
    X = feature_stack.T
    # make the annotation 1-dimensional
    y = annotation.ravel()

    # remove all pixels from the feature and annotations which have
    # not been annotated
    mask = y > 0
    X = X[mask]
    y = y[mask]

    return X, y


def GetTranswellData(base_dir, competition, files_are_in):
    different_assays_folders = [
        file for file in os.listdir(base_dir + competition + files_are_in) if file.endswith(".tif")
    ]

    # get all files that start with WT_
    WT_files = [file for file in different_assays_folders if file.startswith("WT_")]

    Mutant_files = [file for file in different_assays_folders if not file.startswith("WT_")]

    # sort the files
    WT_files.sort()
    Mutant_files.sort()

    # check if a log.txt exists in the base directory
    mode = "w"
    if os.path.exists(base_dir + "log.txt"):
        mode = "a"

    # get a time stamp
    import datetime

    # if the files are empty, write a log into the base directory
    if len(WT_files) == 0:
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")

        with open(base_dir + "log.txt", mode) as f:
            f.write(
                now
                + "\n"
                + "No files with the name <WT_> in the beginning found in the folder: "
                + base_dir
                + competition
                + files_are_in
                # add a new line
                + "\n"
            )
    # if the Mutant files are empty, write a log into the base directory
    # make sure to now overwrite the log
    elif len(Mutant_files) == 0:
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M:%S")
        with open(base_dir + "log.txt", mode) as f:
            f.write(
                now
                + "\n"
                + "No files with NOT the name <WT_> found in the folder: "
                + base_dir
                + competition
                + files_are_in
                + "\n"
            )

    return WT_files, Mutant_files


def GetNormalizationFactor(intensity_img):
    """
    Probably the best strategy is to get the intensity
    of a single cell and then define
    everything as multiple of it.

    Since we do not have the resolution of this with the RFC yet,
    I use another approach based
    on the maximum intensities of the image.
    """

    return np.median(np.sort(intensity_img.ravel())[-1000:])


def MakeSameSizeArray(arr_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    This function takes a list of numpy arrays and makes them the same size by
    appending nan elements
    """
    max_len = max(len(arr) for arr in arr_list)

    new_arr_list = []
    for arr in arr_list:
        if len(arr) < max_len:
            missing_rows = max_len - len(arr)
            new_arr = np.concatenate([arr, np.full(missing_rows, np.nan)])
        else:
            new_arr = arr.copy()

        new_arr_list.append(new_arr)

    return new_arr_list


def GetCompetitionFolders(base_dir):
    """
    This function returns all folders in the base_dir that contain the
    competition data.
    """
    subdirectories = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
    ]
    subdirectories.sort()

    return subdirectories
