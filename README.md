# Competition Assay

Hi Lucas,

I divided this script in several sub scripts with the idea that you can try step by step what you need and what makes sense for you.

Once you are happy with the workflow, I can merge everything together so that it works fully automatically.

## Installation
- Please download the repository and unpack it whereever you feel comfortable (e.g. in your Workspace or on the Desktop)
- Install Anaconda or Miniconda 
- Create a python environment for your project
```bash
conda create -n competition_env
```
- open the environment
```bash
conda activate competition_env
```
- install all necessary packages **manually**
- TODO Eric: create an environment.yml file for: conda env create -f environment.yml
 

## How to organize your data and the scripts
- **this you should already have:** store your data in one folder (e.g. a folder called "dataset_competition_assays") with subfolders (e.g. "competition_2_WTmScarlet_dwspFmNeonGreen") and put inside the "inoculum" and "TW_growth" data
- place the folder in whatever location you like, but the easiest is to place it inside the repository which you just downloaded


## How to run the program

### Step 1 - Denoising
- The first part is denoising using noise2void (since this step needs a tensorflow installation, it is the easiest to run it in google colab)
- create a google drive account if you dont have it yet
- save your data there
- save the model there (this file is too large to be saved on github - you have it already in dropbox under 
imaging_datasets/models/ , but it is better placed in Google Drive where we need it ^^)
- save the colab script in drive too ("noise2void_prediction.ipynb")
- now the folder structure is like:
```
- main folder
--- noise2void_prediction.ipynb
--- models/
    --- n2v_transwell
--- dataset_competition_assays/
    --- competition_2_WTmScarlet_dwspFmNeonGreen/
        --- inoculum/
        --- TW_growth/
```
- run the "noise2void_prediction" script in google colab (it will create a folder called "denoised" (inside the TW_growth) in which all results are saved)
- now download the complete "dataset_competition_assays" to have it locally on your computer (or if you have it already then download only the "denoised" folder and add it to the TW_growth folder)
- P.S. in future we will either do everything in colab or install tensorflow on your computer

### Step 2 - Segmentation
- place the denoised files in the right location (see below)
```
- main folder
--- noise2void_prediction.ipynb
--- models/
    --- RandomForestClassifier_transwell
        --- transwell_denoised_2_categories_Mutant.pkl
        --- transwell_denoised_2_categories_WT.pkl
    --- n2v_transwell
--- dataset_competition_assays/
    --- competition_2_WTmScarlet_dwspFmNeonGreen/
        --- inoculum/
        --- TW_growth/
            --- denoised/
```

- the RandomForestClassifier model is on GitHub, please place it in the models folder (next to the noise2void model)
- open the Segmentation.py script and define the right base directory and model path 
- define the minimum object size (in pixels)
- run the script (dependencies should be already installed)
```
pip3 install scikit-image
or
conda install scikit-image
```
in case skimage is not installed yet    
- the script will create a folder called "segmented" (inside the TW_growth) in which all results are saved (the segmentation is binary, but it can be also a multi-class segmentation, when you set return_binary=False, but this is something that should not be necessary for you)
```python
 Mutant_segmented = helpers.RandomForestSegmentation(
        Mutant_denoised, modelpath_Mutant, return_binary=False, visualize=False
    )
```

### Step 3 - Quantification
There are different scripts for quantification. These are **Area_competition.py**, **Biofilm_Identification.py**,**Local_Competition.py**

#### Area_competition.py
- install polars
```
conda install polars
or
pip install polars
```
- define the base directory and run the script
- results are saved in "Competition_results/" and they are a) csv file for the competition b) histogram of the area distribution of single images c) histogram of the area distribution of all images together TODO: d) for b and c corresponding csv files

#### Biofilm_Identification.py
- define base direcotry and run the script
- results are also saved in "Competition_results/" and they are scatter plots of WT and Mutant for Area of clusters and Intensity (normalized with the median of the highest 100 values (**NOTE**: this leads to 50 maximum values that are above 1 - but mostly only slightly above 1. Taking the maximum value for normalization would be a too noisy measure). TODO: normalize with the intensity of a single cell, if segmentation is able to detect single cells)

#### Local_Competition.py
- define base directory and run the script
- the output is a csv file (and corresponding box plot) that we have not discussed it yet, so I will save it in an extra folder called "local_competition/"
- the local competition describes how much competition has X with Y in the region in which X it colonized, and vice versa. This is a competition based on area and also on intensity density (average intensity in this area)

![Local Competition example](media/example_local_competition.png)

The output can be read as:
- From all the area in which the Mutant is colonized, it is sharing (co-colinized) 60 percent with the WT. The rest of the area (40 percent) is not colonized by the WT, so there is no competition
- From all the area in which the WT is colonized, it is sharing (co-colinized) 30 percent with the Mutant. The rest of the area (70 percent) is not colonized by the Mutant, so there is no competition

- the 2 intensity plots must be changed probably, but they would be interpreted like:
- In the area in which the Mutant is colonized, the WT is not very dense (it has an intensity of around 40 percent of its maximum intensity) 

#TODO: subtract background intensity first (based on 2-class classification) 
#TODO: as a better intensity parameter: compare ratios between mutant and WT intensity instead of the absolute (normalized) value as it is right now. Then compare these ratios with the ones from the distinct area (only X or only Y), to have a baseline


P.S. the covered area by mutant and WT is already calculated and saved in the folder "Competition_results" (in the csv file called "area_covered.csv")

***
### Author
***
Eric Schmidt