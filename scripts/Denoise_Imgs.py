"""
Author: Eric Schmidt

This script is made to denoise images with a pretrained n2v model.
The training of the model is done in the google colab notebook.
"""

import os
from skimage.io import imread, imsave
from CompetitionAssay.denoising import n2vDenoising
from CompetitionAssay.datahandling import GetTranswellData


def main():
    # define base directory
    base_dir = "example_data/competitions/"
    files_are_in = "/TW_growth/"

    # n2v model directory
    model_name = "n2v_transwell"
    model_dir = "models"

    myModel = n2vDenoising(model_name=model_name, model_dir=model_dir)

    # get all the files in the directory
    different_competitions_folders = os.listdir(base_dir)

    for competition in different_competitions_folders:
        output_dir_denoised = base_dir + competition + files_are_in + "denoised/"

        # create output directory if it does not exist
        if not os.path.exists(output_dir_denoised):
            os.makedirs(output_dir_denoised)

        WT_files, Mutant_files = GetTranswellData(base_dir, competition, files_are_in)

        for WT_file, Mutant_file in zip(WT_files, Mutant_files):
            print(WT_file, Mutant_file)

            WT_img = imread(base_dir + competition + files_are_in + WT_file)
            Mutant_img = imread(base_dir + competition + files_are_in + Mutant_file)

            WT_denoised = myModel.predict(WT_img)
            del WT_img
            imsave(output_dir_denoised + WT_file, WT_denoised)
            del WT_denoised

            Mutant_denoised = myModel.predict(Mutant_img)
            del Mutant_img
            imsave(output_dir_denoised + Mutant_file, Mutant_denoised)
            del Mutant_denoised


if __name__ == "__main__":
    main()
