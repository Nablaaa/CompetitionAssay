import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, remove_small_objects

from n2v.models import N2V


def n2vDenoising(img, visualize=False):
    # We import all our dependencies.
    model_name = "n2v_transwell"
    model_dir = "models"
    model = N2V(config=None, name=model_name, basedir=model_dir)

    # change shape to (y,x, c=1)
    img = np.expand_dims(img, axis=-1)

    pred = model.predict(img, axes="YXC")

    if visualize:
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))

        axs[0].imshow(img)
        axs[0].set_title("input")

        axs[1].imshow(pred[:, :, 0])
        axs[1].set_title("output")

        plt.show()

    return pred[:, :, 0]  # return same shape as input


def ReadData(folder, format="tif"):
    """Reads all files of a given format in a folder and returns a list of
    arrays. The format can be any of the formats supported by the Pillow
    library (e.g. 'tif', 'png', 'jpg', etc.)."""
    # Get all files in the folder
    files = os.listdir(folder)
    # Filter files by format
    files = [f for f in files if f.endswith(format)]
    # Read files
    data = []
    for f in files:
        img = Image.open(os.path.join(folder, f))
        data.append(np.array(img))
    return data


def NormalizeImg(img):
    """Normalize img input by subtracting the mean
    and dividing by std"""
    img = img.astype("float32")
    img -= np.mean(img)
    img /= np.std(img)

    return img


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


def GetTranswellData(base_dir, competition, files_are_in):
    different_assays_folders = [
        file
        for file in os.listdir(base_dir + competition + files_are_in)
        if file.endswith(".tif")
    ]

    # get all files that start with WT_
    WT_files = [file for file in different_assays_folders if file.startswith("WT_")]

    Mutant_files = [
        file for file in different_assays_folders if not file.startswith("WT_")
    ]

    # sort the files
    WT_files.sort()
    Mutant_files.sort()

    return WT_files, Mutant_files


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
