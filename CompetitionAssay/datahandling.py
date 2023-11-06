"""
Author: Eric Schmidt

This file contains all classes that help to load, save
and in general handle the data.
"""

import numpy as np
from skimage import filters
import os
from typing import List


# def ReadData(folder, format="tif"):
#     """Reads all files of a given format in a folder and returns a list of
#     arrays. The format can be any of the formats supported by the Pillow
#     library (e.g. 'tif', 'png', 'jpg', etc.)."""
#     # Get all files in the folder
#     files = os.listdir(folder)
#     # Filter files by format
#     files = [f for f in files if f.endswith(format)]
#     # Read files
#     data = []
#     for f in files:
#         img = Image.open(os.path.join(folder, f))
#         data.append(np.array(img))
#     return data


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