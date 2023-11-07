"""
Author: Eric Schmidt

If you want to train your own RFC, then you can use the documentation of scikit-learn:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
or you can use this script. It is a little service for you :)

You can try it with the given GT in the example_data folder, but if you want to reach better results,
you should create your own GT.(I recommend using napari for this, e.g. RFC_Draw_GT.ipynb in this
folder)
"""

from skimage.io import imread
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from CompetitionAssay.datahandling import generate_feature_stack, format_data
import pickle
import time


def Train_RFC(
    denoised_img_filename: str,
    manual_labels_filename: str,
    model_path: str,
    model_name: str,
    param_grid: dict,
):
    img = imread(denoised_img_filename)
    GT = imread(manual_labels_filename)

    feature_stack = generate_feature_stack(img)

    # train classifier if not trained yet
    classifier = RandomForestClassifier()

    X, y = format_data(feature_stack, GT)

    # perform grid search
    grid_search = GridSearchCV(classifier, param_grid, cv=5)
    grid_search.fit(X, y)  # X and y are your training data and labels, respectively

    # print the results
    results = grid_search.cv_results_
    scores = np.array(results["mean_test_score"]).reshape(
        len(param_grid["n_estimators"]), len(param_grid["max_depth"])
    )
    print(scores)

    best_classifier = grid_search.best_estimator_

    # save the best classifier
    timestr = time.strftime("%Y%m%d-%H%M%S")
    pickle.dump(best_classifier, open(model_path + timestr + "_" + model_name, "wb"))


def main():
    print(__doc__)

    # define path to input img and GT
    img_path = "example_data/RFC_Trainingsdata/Mutant/denoised_img/"
    img_fn = "dwspF_C1-MAX_20230424_5hpif_mix2_WTmScarlet_dwspFmNeonGreen_ours_R3_003-1.tif"
    denoised_img_filename = img_path + img_fn

    GT_path = "example_data/RFC_Trainingsdata/Mutant/GT/"
    GT_fn = "dwspF_C1-MAX_20230424_5hpif_mix2_WTmScarlet_dwspFmNeonGreen_ours_R3_003-1.tif"
    manual_labels_filename = GT_path + GT_fn

    # define model name and directory
    model_path = "models/RandomForestClassifier_transwell/"
    model_name = "myRFC_Model.pkl"

    # define parameter grid
    param_grid = {
        "n_estimators": [50, 100],  # Vary the number of trees
        "max_depth": [2, 3],  # Vary the maximum depth of trees
    }

    Train_RFC(denoised_img_filename, manual_labels_filename, model_path, model_name, param_grid)


if __name__ == "__main__":
    main()
