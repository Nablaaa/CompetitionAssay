"""
author: Eric Schmidt

This is an example script for the use of the Random Forest Classifier (RFC)

This script is used to binarize the denoised competition assay files.
The RFC is trained to predict background, foreground and blurry rest.
Only the foreground and background will be used as the final binary output.

Please define the path to the denoised competition assay to run the file
"""


from skimage.io import imread
from CompetitionAssay.binarization import myRFC


def main():
    print(__doc__)

    denoised_img_path = "examples/example_data/competition_2_WTmScarlet_dwspFmNeonGreen/TW_growth/denoised/WT_C2-MAX_20230424_5hpif_mix2_WTmScarlet_dwspFmNeonGreen_ours_R3_003-1.tif"
    model_path = "models/RandomForestClassifier_transwell/transwell_denoised_2_categories_WT.pkl"
    WT = imread(denoised_img_path)

    # use the RFC to predict the binary image
    RFC = myRFC(model_path)
    binary = RFC.predict(WT)
    RFC.visualize()

    # save the binary image
    # from skimage.io import imsave
    # imsave("binary.tif", binary)


if __name__ == "__main__":
    main()
