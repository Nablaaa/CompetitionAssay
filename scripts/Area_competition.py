"""
Author: Eric Schmidt

In this script the used area of the cells is calculated to compare
which cell type spreads more.

Results seem to make sense with respect to images
"""

import os
from skimage.io import imread
import numpy as np
import polars as pl
from matplotlib import pyplot as plt
from CompetitionAssay import datahandling, quantification, visualization


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

        # get all the intensity files in the directory
        WT_files, Mutant_files = datahandling.GetTranswellData(base_dir, competition, files_are_in)
        WT_binary, Mutant_binary = datahandling.GetTranswellData(
            base_dir, competition, files_are_in + "binary/"
        )

        # TODO: there must be a better way to get this name
        repetition_name = [WT_binary[i][-12:-4] for i in range(len(WT_binary))]

        area_covered_WT = []
        area_covered_Mutant = []

        all_single_areas_WT = []
        all_single_areas_Mutant = []

        WT_mean_size, WT_median_size, WT_std_size = [], [], []
        Mutant_mean_size, Mutant_median_size, Mutant_std_size = [], [], []

        all_single_cell_intensities_WT = []
        all_single_cell_intensities_Mutant = []

        WT_mean_intensity, WT_median_intensity, WT_std_intensity = [], [], []
        Mutant_mean_intensity, Mutant_median_intensity, Mutant_std_intensity = [], [], []

        # go through all the files
        for i, (WT_file, Mutant_file, WT_binary_file, Mutant_binary_file) in enumerate(
            zip(WT_files, Mutant_files, WT_binary, Mutant_binary)
        ):
            WT_img = imread(base_dir + competition + files_are_in + WT_file)
            Mutant_img = imread(base_dir + competition + files_are_in + Mutant_file)

            WT_binary_img = imread(
                base_dir + competition + files_are_in + "binary/" + WT_binary_file
            )
            Mutant_binary_img = imread(
                base_dir + competition + files_are_in + "binary/" + Mutant_binary_file
            )

            # calculate the area of the cells
            WT_area_per = quantification.GetCoveredAreaPercent(WT_binary_img)
            Mutant_area_per = quantification.GetCoveredAreaPercent(Mutant_binary_img)

            area_covered_WT.append(WT_area_per)
            area_covered_Mutant.append(Mutant_area_per)

            # get area distribution of single cells
            (
                WT_single_cell_area,
                WT_single_cell_intensity,
            ) = quantification.GetSingleCellAreaAndIntensity(WT_binary_img, WT_img)
            all_single_areas_WT.append(WT_single_cell_area)
            all_single_cell_intensities_WT.append(WT_single_cell_intensity)

            WT_mean_size.append(np.mean(WT_single_cell_area))
            WT_median_size.append(np.median(WT_single_cell_area))
            WT_std_size.append(np.std(WT_single_cell_area))

            WT_mean_intensity.append(np.mean(WT_single_cell_intensity))
            WT_median_intensity.append(np.median(WT_single_cell_intensity))
            WT_std_intensity.append(np.std(WT_single_cell_intensity))

            (
                Mutant_single_cell_area,
                Mutant_single_cell_intensity,
            ) = quantification.GetSingleCellAreaAndIntensity(Mutant_binary_img, Mutant_img)
            all_single_areas_Mutant.append(Mutant_single_cell_area)
            all_single_cell_intensities_Mutant.append(Mutant_single_cell_intensity)

            Mutant_mean_size.append(np.mean(Mutant_single_cell_area))
            Mutant_median_size.append(np.median(Mutant_single_cell_area))
            Mutant_std_size.append(np.std(Mutant_single_cell_area))

            Mutant_mean_intensity.append(np.mean(Mutant_single_cell_intensity))
            Mutant_median_intensity.append(np.median(Mutant_single_cell_intensity))
            Mutant_std_intensity.append(np.std(Mutant_single_cell_intensity))

            # plot single competition as histogram
            competition_hist_fig = visualization.PlotCompetitionsHistogram(
                WT_single_cell_area, Mutant_single_cell_area, visualize=False
            )

            save_with_repetition_name = WT_binary[i][-12:-4]

            competition_hist_fig.savefig(
                output_dir + save_with_repetition_name + "_competition_histogram.png", dpi=300
            )
            plt.close(competition_hist_fig)

            # save the data from the plot as csv
            (
                WT_single_cell_area,
                Mutant_single_cell_area,
                WT_single_cell_intensity,
                Mutant_single_cell_intensity,
            ) = datahandling.MakeSameSizeArray(
                [
                    WT_single_cell_area,
                    Mutant_single_cell_area,
                    WT_single_cell_intensity,
                    Mutant_single_cell_intensity,
                ]
            )

            params = pl.DataFrame(
                {
                    "WT_single_cell_area[px]": WT_single_cell_area,
                    "Mutant_single_cell_area[px]": Mutant_single_cell_area,
                    "WT_single_cell_intensity": WT_single_cell_intensity,
                    "Mutant_single_cell_intensity": Mutant_single_cell_intensity,
                }
            )

            params.write_csv(output_dir + save_with_repetition_name + ".csv", separator=",")

        all_single_areas_WT = np.concatenate([arr.flatten() for arr in all_single_areas_WT])
        all_single_areas_Mutant = np.concatenate([arr.flatten() for arr in all_single_areas_Mutant])

        visualization.Plot_Area_Histogram_Overall(
            all_single_areas_WT, all_single_areas_Mutant, competition, output_dir, visualize=False
        )

        # convert to numpy
        area_covered_WT = np.array(area_covered_WT)
        area_covered_Mutant = np.array(area_covered_Mutant)
        ratio_WT_Mutant = area_covered_WT / area_covered_Mutant

        # create polars df
        df = pl.DataFrame(
            {
                "Name": repetition_name,
                "WT[per]": area_covered_WT,
                "Mutant[per]": area_covered_Mutant,
                "ratio[WT/Mutant]": ratio_WT_Mutant,
                "WT_mean_cluster_size[px]": WT_mean_size,
                "WT_median_cluster_size[px]": WT_median_size,
                "WT_std_cluster_size[px]": WT_std_size,
                "WT_mean_intensity": WT_mean_intensity,
                "WT_median_intensity": WT_median_intensity,
                "WT_std_intensity": WT_std_intensity,
                "Mutant_mean_cluster_size[px]": Mutant_mean_size,
                "Mutant_median_cluster_size[px]": Mutant_median_size,
                "Mutant_std_cluster_size[px]": Mutant_std_size,
                "Mutant_mean_intensity": Mutant_mean_intensity,
                "Mutant_median_intensity": Mutant_median_intensity,
                "Mutant_std_intensity": Mutant_std_intensity,
            }
        )

        # save the polars df
        df.write_csv(output_dir + "area_covered.csv", separator=",")

        print("done")


if __name__ == "__main__":
    main()
