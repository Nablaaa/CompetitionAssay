"""
Author: Eric Schmidt

In this script the used area of the cells is calculated to compare
which cell type spreads more.

Results seem to make sense with respect to images
"""

import os
from skimage.io import imread
from CompetitionAssay.datahandling import GetTranswellData
from CompetitionAssay.quantification import GetSingleCellAreaAndIntensity
from CompetitionAssay.visualization import Plot_Biofilm_Identification


def main():
    # define base directory
    base_dir = "example_data/"
    files_are_in = "/TW_growth/"

    # get all the files in the directory
    different_competitions_folders = os.listdir(base_dir)

    for competition in different_competitions_folders:
        # output directory
        output_dir = base_dir + competition + files_are_in + "Competition_results/"

        # create output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # get all the intensity and binary files in the directory
        WT_files, Mutant_files = GetTranswellData(base_dir, competition, files_are_in)
        WT_binary, Mutant_binary = GetTranswellData(base_dir, competition, files_are_in + "binary/")

        assert len(WT_files) == len(Mutant_files), "WT and Mutant files must have the same length"
        assert len(WT_binary) == len(Mutant_binary), "WT and Mutant binary files must have the same length"
        assert WT_files == WT_binary, "WT and WT binary files must have the same name"
        assert Mutant_files == Mutant_binary, "Mutant and Mutant binary files must have the same name"

        # go through all the files
        for WT_file, Mutant_file, WT_binary_file, Mutant_binary_file in zip(
            WT_files, Mutant_files, WT_binary, Mutant_binary
        ):
            print(WT_file + " is in progress")

            WT_img = imread(base_dir + competition + files_are_in + WT_file)
            Mutant_img = imread(base_dir + competition + files_are_in + Mutant_file)

            WT_binary = imread(base_dir + competition + files_are_in + "binary/" + WT_binary_file)
            Mutant_binary = imread(
                base_dir + competition + files_are_in + "binary/" + Mutant_binary_file
            )

            # get area and intensity of the cells
            WT_area, WT_intensity = GetSingleCellAreaAndIntensity(WT_binary, WT_img)
            Mutant_area, Mutant_intensity = GetSingleCellAreaAndIntensity(Mutant_binary, Mutant_img)

            _, fig = Plot_Biofilm_Identification(
                WT_intensity, Mutant_intensity, WT_area, Mutant_area
            )
                
            fig.savefig(output_dir + WT_file[:-4] + "_intensity_vs_area.png", dpi=500)
            del fig




if __name__ == "__main__":
    main()
