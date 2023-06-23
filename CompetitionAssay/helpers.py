import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, remove_small_objects

# from n2v.models import N2V

from sklearn.ensemble import RandomForestClassifier
import pickle
from skimage import filters


def RemoveSmallObjects(segmentation, min_size=9):
    """
    Remove small objects from a segmentation
    """
    # segmentation can have multiple labels, so work with binary
    binary = np.zeros_like(segmentation)
    binary[segmentation > 0] = 1
    segmentation_without_small_objects = remove_small_objects(
        binary.astype("bool"), min_size=min_size
    )

    # convert back to original labels
    segmentation[segmentation_without_small_objects == 0] = 0
    return segmentation


def PlotCompetitionsHistogram(WT_single_cell_area, Mutant_single_cell_area, visualize=False):
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


def Plot_Area_Histogram(
    all_single_areas_WT, all_single_areas_Mutant, competition, output_dir, visualize=False
):
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


# pick out a certain label (e.g. when classifier has 2 categories)
def PickLabel(img, label):
    img[img != label] = 0
    img[img == label] = 1
    return img


def n2vDenoising(img, visualize=False, pass_it=False):
    if pass_it:
        return img

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


def RandomForestSegmentation(img, modelpath, visualize=False):
    feature_stack = generate_feature_stack(img)
    loaded_classifier = pickle.load(open(modelpath, "rb"))

    result_1d = loaded_classifier.predict(feature_stack.T) - 1  # subtract 1 to make background = 0
    result_2d = result_1d.reshape(img.shape)

    if visualize:
        # create a plot with 2 subplots and add the img and the result_2d
        fig, axs = plt.subplots(1, 2, figsize=(10, 10))
        axs[0].imshow(img)
        axs[0].set_title("input")
        axs[1].imshow(result_2d)
        axs[1].set_title("output")
        plt.show()

    return result_2d


def generate_feature_stack(image):
    # determine features
    blurred = filters.gaussian(image, sigma=2)
    edges = filters.sobel(blurred)

    # collect features in a stack
    # The ravel() function turns a nD image into a 1-D image.
    # We need to use it because scikit-learn expects values in a 1-D format here.
    feature_stack = [image.ravel(), blurred.ravel(), edges.ravel()]

    # return stack as numpy-array
    return np.asarray(feature_stack)


def format_data(feature_stack, annotation):
    # reformat the data to match what scikit-learn expects
    # transpose the feature stack
    X = feature_stack.T
    # make the annotation 1-dimensional
    y = annotation.ravel()

    # remove all pixels from the feature and annotations which have not been annotated
    mask = y > 0
    X = X[mask]
    y = y[mask]

    return X, y


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


def GetCoveredAreaPercent(binary_img):
    return np.sum(binary_img) / (binary_img.shape[0] * binary_img.shape[1])


def GetSingleCellArea(binary_img):
    """Get the area of a single cell in a binary image"""

    ChangeMorphology(binary_img, 2, show=False)

    from skimage.measure import label, regionprops

    # label the image
    labeled_img = label(binary_img)

    # get the area of each object
    props = regionprops(labeled_img)

    # get all areas
    areas = [prop.area for prop in props]

    return np.array(areas)


def GetSingleCellAreaAndIntensity(binary_img, intensity_img):
    """Get the area of a single cell in a binary image"""

    ChangeMorphology(binary_img, 2, show=False)

    from skimage.measure import label, regionprops

    # label the image
    labeled_img = label(binary_img)

    # get the area and intensityof each object
    props = regionprops(labeled_img, intensity_image=intensity_img)

    # get all areas
    areas = [prop.area for prop in props]

    # get all mean intensities
    mean_intensities = [prop.mean_intensity for prop in props]

    return np.array(areas), np.array(mean_intensities)


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
