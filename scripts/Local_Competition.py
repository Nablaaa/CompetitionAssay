"""
Author: Eric Schmidt

In this script I am comparing the area and intensity of WT and Mutant
with respect to the Mutant and WT itself. I define a local competition
based on the area that is covered by WT and Mutant.

So the competition is defined by:


a) 
How much Mutant is found in the area that is covered by WT and vice versa.
(e.g.   Mutant covers only 10 % of the area that is covered by WT
and vice versa 
        WT covers over 90 % of the area that is covered by Mutant)

b)
This consideration makes sense, if the Mutant-WT overlap is large enough. 
(if they are distinct, then it will end up in a comparison between WT/Mutant and background of Mutant/WT)

How bright/dense is the Mutant in the area that is covered by WT and vice versa.
(e.g.   In the area that is covered by WT, the Mutant is per event 10 times brighter then the WT
         ==== with event I mean a intensity per label (so a brightness density ====

and vice versa
        In the area that is covered by Mutant, the WT only reaches 20 % of the brightness of the Mutant)
        
"""


import os
from skimage.io import imread
from CompetitionAssay import datahandling, quantification
import polars as pl
from matplotlib import pyplot as plt


def main():
    # define base directory
    base_dir = "example_data/competitions/"
    files_are_in = "/TW_growth/"

    # get all the files in the directory
    different_competitions_folders = datahandling.GetCompetitionFolders(base_dir)

    for competition in different_competitions_folders:
        # output directory
        output_dir = base_dir + competition + files_are_in + "Competition_results/"

        # create output directory if it does not exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # get all filenames of the intensity files (not the denoised files,
        # since noise2void may manipulates the intensity so that it can not
        # be used for quantification)
        WT_files, Mutant_files = datahandling.GetTranswellData(base_dir, competition, files_are_in)

        # get binary files
        WT_binary_files, Mutant_binary_files = datahandling.GetTranswellData(
            base_dir, competition, files_are_in + "binary/"
        )

        assert (
            len(WT_files) == len(Mutant_files) == len(WT_binary_files) == len(Mutant_binary_files)
        ), "WT and Mutant files must have the same length"

        assert WT_files == WT_binary_files, "WT and WT binary files must have the same name"
        assert (
            Mutant_files == Mutant_binary_files
        ), "Mutant and Mutant binary files must have the same name"

        repetition_name = [WT_binary_files[i][-12:-4] for i in range(len(WT_binary_files))]
        list_area_WT_in_Mutant = []
        list_area_Mutant_in_WT = []
        list_intensity_density_WT_in_Mutant = []
        list_intensity_density_Mutant_in_WT = []
        list_intensity_density_WT_alone = []
        list_intensity_density_Mutant_alone = []
        list_intensity_density_WT_general = []
        list_intensity_density_Mutant_general = []

        df = pl.DataFrame(
            {
                "Name": [],
                "area_WT_in_Mutant[per]": [],
                "Norm_intensity_density_WT_in_Mutant": [],
                "area_Mutant_in_WT[per]": [],
                "Norm_intensity_density_Mutant_in_WT": [],
            }
        )

        # go through all the files
        for i, (WT_file, Mutant_file, WT_binary_file, Mutant_binary_file) in enumerate(
            zip(WT_files, Mutant_files, WT_binary_files, Mutant_binary_files)
        ):
            # load the images
            WT_img = imread(base_dir + competition + files_are_in + WT_file)
            Mutant_img = imread(base_dir + competition + files_are_in + Mutant_file)
            WT_binary = imread(base_dir + competition + files_are_in + "binary/" + WT_binary_file)
            Mutant_binary = imread(
                base_dir + competition + files_are_in + "binary/" + Mutant_binary_file
            )

            # perform local competition on area and intensity
            (
                area_competition_x_in_y,
                area_competition_y_in_x,
                normalized_intensity_density_x,
                normalized_intensity_density_y,
                normalized_intensity_density_x_alone,
                normalized_intensity_density_y_alone,
                normalized_intensity_density_x_total,
                normalized_intensity_density_y_total,
            ) = quantification.LocalCompetition(
                WT_img, WT_binary, Mutant_img, Mutant_binary, visualize=False
            )

            # append the results to the lists
            list_area_WT_in_Mutant.append(area_competition_x_in_y)
            list_area_Mutant_in_WT.append(area_competition_y_in_x)
            list_intensity_density_WT_in_Mutant.append(normalized_intensity_density_x)
            list_intensity_density_Mutant_in_WT.append(normalized_intensity_density_y)
            list_intensity_density_WT_alone.append(normalized_intensity_density_x_alone)
            list_intensity_density_Mutant_alone.append(normalized_intensity_density_y_alone)
            list_intensity_density_WT_general.append(normalized_intensity_density_x_total)
            list_intensity_density_Mutant_general.append(normalized_intensity_density_y_total)

        # add the results to the dataframe
        df = pl.from_dict(
            {
                "Name": repetition_name,
                "area_WT_in_Mutant[per]": list_area_WT_in_Mutant,
                "area_Mutant_in_WT[per]": list_area_Mutant_in_WT,
                "Norm_intensity_density_WT_general": list_intensity_density_WT_general,
                "Norm_intensity_density_WT_in_Mutant": list_intensity_density_WT_in_Mutant,
                "Norm_intensity_density_WT_alone": list_intensity_density_WT_alone,
                "Norm_intensity_density_Mutant_general": list_intensity_density_Mutant_general,
                "Norm_intensity_density_Mutant_in_WT": list_intensity_density_Mutant_in_WT,
                "Norm_intensity_density_Mutant_alone": list_intensity_density_Mutant_alone,
            }
        )

        # save the polars df
        df.write_csv(output_dir + "local_competition.csv", separator=",")

        # create a boxplot of the results
        plt.figure(figsize=(15, 15))

        df.to_pandas().boxplot()
        # rotate the x labels by 45 degrees
        plt.xticks(rotation=30)
        plt.title("Local Competition")
        plt.tight_layout()
        plt.savefig(output_dir + "boxplot_local_competition.png", dpi=500)
        plt.close()


if __name__ == "__main__":
    main()
