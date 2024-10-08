"""
Author: Eric Schmidt

This script is used to test the segmentation results
to check, if the segmentation makes sense
For this, the original images are loaded and the
segmentation is overlayed
The results are saved in a folder as png so that one can
easily click through them
"""

import os
from skimage.io import imread
from CompetitionAssay import datahandling, visualization
import matplotlib.pyplot as plt

def main():
    # define base directory
    base_dir = "example_data/competitions/"
    files_are_in = "/TW_growth/"

    different_competitions_folders = datahandling.GetCompetitionFolders(base_dir)

    for competition in different_competitions_folders:
        print(competition)
        # output directory
        output_dir = base_dir + competition + files_are_in + "RFC_output/"

        # create output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        WT_files, Mutant_files = datahandling.GetTranswellData(base_dir, competition, files_are_in)
        WT_binary, Mutant_binary = datahandling.GetTranswellData(
            base_dir, competition, files_are_in + "binary/"
        )

        # go through all the files
        for WT_file, Mutant_file, WT_binary_file, Mutant_binary_file in zip(
            WT_files, Mutant_files, WT_binary, Mutant_binary
        ):
            WT_img = imread(base_dir + competition + files_are_in + WT_file)
            Mutant_img = imread(base_dir + competition + files_are_in + Mutant_file)

            WT_binary = imread(base_dir + competition + files_are_in + "binary/" + WT_binary_file)
            Mutant_binary = imread(
                base_dir + competition + files_are_in + "binary/" + Mutant_binary_file
            )

            ax, merged_plot = visualization.OverlaySegmentation(
                WT_img, WT_binary, Mutant_img, Mutant_binary
            )

            repetition_name = WT_file[-12:-4]

            merged_plot.savefig(output_dir + repetition_name + "_visualization.png", dpi=300)

            # close figure
            plt.close('all')

            del ax
            del merged_plot
            del WT_binary, Mutant_binary, WT_img, Mutant_img


if __name__ == "__main__":
    main()
