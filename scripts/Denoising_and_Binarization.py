"""
Author: Eric Schmidt



This script gives an example of how to set up the RFC in a workflow that loads WT and mutant at the same time.

The input is the model path, as well as the folder that consists the competition assay files.

Later it will be integrated in a workflow (Quantification_transwell.py)
"""

import os
from skimage.io import imread, imsave
from CompetitionAssay.datahandling import GetTranswellData, GetCompetitionFolders
from CompetitionAssay.binarization import myRFC, RemoveSmallObjects


def main():
    # define base directory
    base_dir = "example_data/competitions/"
    files_are_in = "/TW_growth/"

    RFC_modelpath_WT = (
        "models/RandomForestClassifier_transwell/transwell_denoised_2_categories_WT.pkl"
    )
    RFC_modelpath_Mutant = (
        "models/RandomForestClassifier_transwell/transwell_denoised_2_categories_Mutant.pkl"
    )

    # objects have to be at least 49 objects (e.g. 7x7) large
    min_size = 49  # pixels

    perform_denoising = True
    if perform_denoising:
        from CompetitionAssay.denoising import n2vDenoising

        # n2v model directory
        model_name = "n2v_transwell"
        model_dir = "models"

        myModel = n2vDenoising(model_name=model_name, model_dir=model_dir)

    # get all the files in the directory
    different_competitions_folders = GetCompetitionFolders(base_dir)

    for competition in different_competitions_folders:
        output_dir_binary = base_dir + competition + files_are_in + "binary/"
        output_dir_denoised = base_dir + competition + files_are_in + "denoised/"

        # create output directory if it does not exist
        if not os.path.exists(output_dir_binary):
            os.makedirs(output_dir_binary)

        # create output directory if it does not exist
        if not os.path.exists(output_dir_denoised):
            os.makedirs(output_dir_denoised)

        WT_files, Mutant_files = GetTranswellData(base_dir, competition, files_are_in)

        for WT_file, Mutant_file in zip(WT_files, Mutant_files):
            print(WT_file, Mutant_file)

            WT_img = imread(base_dir + competition + files_are_in + WT_file)
            Mutant_img = imread(base_dir + competition + files_are_in + Mutant_file)

            if perform_denoising:
                # start here the denoising with a pretrained model
                WT_denoised = myModel.predict(WT_img)
                del WT_img
                imsave(output_dir_denoised + WT_file, WT_denoised)

                Mutant_denoised = myModel.predict(Mutant_img)
                del Mutant_img
                imsave(output_dir_denoised + Mutant_file, Mutant_denoised)

            else:
                assert os.path.exists(
                    base_dir + competition + files_are_in + "denoised/" + WT_file
                ), (
                    "Please add a correct path to the denoised files, your current path is: "
                    + base_dir
                    + competition
                    + files_are_in
                    + "denoised/"
                    + WT_file
                    + " and it does not exist"
                )

                WT_denoised = imread(base_dir + competition + files_are_in + "denoised/" + WT_file)
                Mutant_denoised = imread(
                    base_dir + competition + files_are_in + "denoised/" + Mutant_file
                )

            # start here the segmentation with a pretrained model

            WT_RFC = myRFC(RFC_modelpath_WT)
            WT_binary = WT_RFC.predict(WT_denoised)
            del WT_denoised
            WT_without_small_objects = RemoveSmallObjects(WT_binary, min_size=min_size)
            del WT_binary
            imsave(output_dir_binary + WT_file, WT_without_small_objects)
            del WT_without_small_objects

            mutant_RFC = myRFC(RFC_modelpath_Mutant)
            Mutant_binary = mutant_RFC.predict(Mutant_denoised)
            del Mutant_denoised
            Mutant_without_small_objects = RemoveSmallObjects(Mutant_binary, min_size=min_size)
            del Mutant_binary
            imsave(output_dir_binary + Mutant_file, Mutant_without_small_objects)
            del Mutant_without_small_objects


if __name__ == "__main__":
    main()
