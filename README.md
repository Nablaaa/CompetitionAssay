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
- The first part is denoising using noise2void (since this step needs a tensorflow installation, it is the easiest to run it in google colab)
- create a google drive account if you dont have it yet
- save your data there
- save the model there (this file is too large to be saved on github - you have it already in dropbox under 
imaging_datasets/models/ , but it is better placed in Google Drive where we need it ^^)
- save the colab script in drive too ("noise2void_prediction.ipynb")
- run the script in google colab (it will create a folder called "denoised" in which all results are saved)
- download the results and add it to your data on your computer (so that you have the folder structure like: dataset_competition_assays/competition_2_WTmScarlet_dwspFmNeonGreen/TW_growth/denoised)
- in future we will either do everything in colab or install tensorflow on your computer






***
### Author
***
Eric Schmidt
