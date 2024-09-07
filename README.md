# Competition Assay


## Introduction
This repository contains scripts to analyze competition assays. The scripts are used by the paper: `Pseudomonas aeruginosa faces a fitness trade-off between mucosal colonization and antibiotic tolerance during airway infections` <br>

***

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [How to organize your data and the scripts](#how-to-organize-your-data-and-the-scripts)
4. [How to run the program](#how-to-run-the-program)
    1. [Step 1 - Denoising](#step-1---denoising)
        1. [Google Colab (recommended in case you should have problems with tensorflow on your machine)](#google-colab-recommended-in-case-you-should-have-problems-with-tensorflow-on-your-machine)
    2. [Step 2 - Segmentation/Binarization](#step-2---segmentationbinarization)
    3. [Step 3 - Quantification](#step-3---quantification)
        1. [Area_competition.py](#areacompetitionpy)
        2. [Local_Competition.py](#localcompetitionpy)
        3. [Biofilm_Identification.py](#biofilmidentificationpy)
5. [How to train the model yourself](#how-to-train-the-model-yourself)
    1. [noise2void resources](#noise2void-resources)
    2. [RandomForestClassifier](#randomforestclassifier)
6. [FAQ](#faq)



## Installation

### For Anaconda/Miniconda Users that want to run everything locally:
- create and activate an environment that has python 3.9 
```bash
conda create -n competition_env python=3.9
conda activate competition_env
```

Inside the activated environment, type:
```bash
pip install competitionassay
```
to download all necessary dependencies. <br>
If this does not work, please take a look at the [FAQ](#faq) section.




## How to organize your data and the scripts
- take a look at the structure in the "examples" folder. This is how your data should be organized. So the structure is, to have a base directory (e.g. example_data) and inside you have it organized like:

```
-example_data
|----competitions
|    |   
|    |---competition_2_WTmScarlet_dwspFmNeonGreen
|    |   |
|    |   |--TW_growth
|    |      |   WT_name_repetition_1.tif
|    |      |   WT_name_repetition_2.tif
|    |      |   WT_name_repetition_3.tif
|    |      |   ...
|    |      |   Mutant_name_repetition_1.tif
|    |      |   Mutant_name_repetition_2.tif
|    |      |   Mutant_name_repetition_3.tif
|    |      |   ...
|    |---other_competition_folders
|    |   
|    |   |--TW_growth
|    |       |   ...
|    |
|----other folders
```

## How to run the program
- download the data, the models (noise2void and RandomForestClassifier) and the scripts from the repository. (or the complete repository)
- open your terminal and navigate to the directory
- open the scripts which you would like to test (e.g. start with Denoise_Imgs.py) and modify the base directory and the model path (if necessary)
- then run the scripts from your terminal or from your IDE (e.g. VSCode)

```
conda activate competition_env
cd /path/to/your/CompetitionAssay

python3 scripts/Denoise_Imgs.py
python3 scripts/Denoising_and_Binarization.py
python3 scripts/...
```


### Step 1 - Denoising
The first part is denoising using **noise2void** ([GitHub](https://github.com/juglab/n2v), [paper](https://arxiv.org/abs/1811.10980)). This can be done on your machine with the Denoise_Imgs.py script, but also on Google Colab with the example script "noise2void_prediction.ipynb". The advantage of Google Colab is that you do not need to install tensorflow on your computer and you can use the GPU of Google Colab for the denoising. The disadvantage is that you need to upload your data to Google Drive and download it again after the denoising.

For making the predictions, it is necessary to load the pretrained model. This can be found under the following link: TODO: [add here a link to the model - responsibility of the lab]

The denoised files are automatically saved in a folder together with your files. The folder is called "denoised".



#### Google Colab (recommended in case you should have problems with tensorflow on your machine)
- create a google drive account if you dont have it yet
- create a main folder and this file structure:

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

- save your data there
- save the model there 
- save the colab script ([noise2void_prediction.ipynb](scripts/noise2void_prediction.ipynb)) there - this script can be found in the repository
- run the "noise2void_prediction.ipynb" script in google colab (it will create a folder called "denoised" (inside the TW_growth) in which all results are saved)
- now download the complete "dataset_competition_assays" to have it locally on your computer (or if you have it already on your computer, then download only the "denoised" folder and add it to the TW_growth folder)


### Step 2 - Segmentation/Binarization

The denoised files are used for binarization via the [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html). The RFC outputs 3 labels (1=background, 2=cell, 3=blurry regions), which will be reduced to a binary image (0=background, 1=cell). To run this script, open the ["Denoising_and_Binarization.py"](scripts/Denoising_and_Binarization.py) script and set the correct directories to the data and to the model (stored under ["models/RandomForestClassifier_transwell/"](models/RandomForestClassifier_transwell)). Also, if you have not done the denoising yet, set the parameter "perform_denoising" to True. In this case, please also direct to the "model_name" and "model_dir" of the [noise2void model](models/n2v_transwell/).

Once you run the script, a folder called "binary" will be created. This folder consist of all binary images that are produced by the RFC. If you want to visualize the results as image-mask overlay, then please use the script [Visualize_RFC_Results.py](scripts/Visualize_RFC_Results.py). This script will create a folder called "RFC_output" in which all overlays are saved. 



### Step 3 - Quantification
There are different scripts for quantification. Not all of them ended up in the paper. The scripts are [Area_competition.py](scripts/Area_competition.py), [Local_Competition.py](scripts/Local_Competition.py), [Biofilm_Identification.py](scripts/Biofilm_Identification.py)

#### Area_competition.py
This script measures for every competition the covered areas by the WT and the competing Mutant and also the corresponding intensity. The output are several csv files and plots. Every competitionassay consist of several repetitions of the same competition (e.g. WT against wsp, Repetition 1, Repetition 2, ...). Every repetition (WT, Mutant) consist of several clusters (you can compare it with the results from [Step 2](#step-2---segmentationbinarization) when using "Visualize_RFC_Results.py"). These clustered are labeled and every cluster results in an area and an intensity.These values are saved as csv file. The files have a name like "R2_002-1.csv". The corresponding area plot is called "name_competition_histogram.png".

Finally, a summary of all repetitions is saved as csv file with the name "area_covered.csv" and the corresponding plot is called "area_covered.png".

#### Local_Competition.py
This script is now taking a closer look into the actual competitions. It is detecting the regions, in which a cluster has overlap with another cluster (competition) and in which regions the clusters are alone (no competition). The script returns one csv file and one figure per competition. The csv file contains of the following parameters: <br>

**Name**

**Area WT in Mutant (in percent)**
- it describes how much area, in which the mutant spreads, is also covered by the WT. In this area, the Mutant competes with the WT. (1- Area WT in Mutant) is the percentage of area in which the Mutant is without competition

**Area Mutant in WT (in percent)**
- it is the same logic as above

**Norm_intensity_density_WT_in_Mutant**
- it is a measure for the brightness of the clusters from the WT that is in the region, covered by the mutant
- the brightness is normalized by approximately the maximum of the intensity from the WT (details in [datahandling.py](CompetitionAssay/datahandling.py))
- this parameter can be used to compare the intensity of WT in the region of the Mutant with the intensity of WT in the region where the WT is alone (no competition)

**Norm_intensity_density_WT_alone**
- this parameter is the same as above, but it is the intensity of the WT in the region where the WT is alone (no competition, but also no protection from e.g. antibiotics)

**Norm_intensity_density_WT_general**
- it is a measure for the brightness of the clusters from the WT in general. This parameter is helpful to interpret the results of the normalized intensity, because one can compare the Norm_intensity_density_WT_in_Mutant and the Norm_intensity_density_WT_alone, with the Norm_intensity_density_WT_general to see, if the intensity increases or decreases for the WT, when being in the region of the mutant.



![Local Competition example](media/example_local_competition.png)

The output can be read as:

**Area**
- From all the area in which the Mutant is colonized, it is sharing (co-colinized) 60 percent with the WT. The rest of the area (40 percent) is not colonized by the WT, so the mutant is alone there (no competition)
- From all the area in which the WT is colonized, it is sharing (co-colinized) 30 percent with the Mutant. The rest of the area (70 percent) is not colonized by the Mutant, so the WT is alone there (no competition)


**Intensity**
- the normalized intensity of the WT in general is 0.4, so in average it is 40 percent of the highest intensities of the WT (I did not normalize with the maximum intensity, since it is not resistant to outliers)
- the normalized intensity of the WT in the region of the Mutant is 0.45, so a bit higher than the average
- the normalized intensity of the WT in the region where the WT is alone is 0.4, so a bit lower than average and lower than the area with overlap.
- the same principal applies for the intensities of the mutant

So for example, if you want to test, if the WT is more resistant to antibiotics in the region of the Mutant, then you can compare the normalized intensity of the WT in the region of the Mutant with the normalized intensity of the WT in the region where the WT is alone. If the normalized intensity of the WT in the region of the Mutant is higher than the normalized intensity of the WT in the region where the WT is alone, then the WT is more resistant to antibiotics in the region of the Mutant (e.g. because it is protected by the biofilm). 




#### Biofilm_Identification.py
This script is not used in the paper, but it might help to define biofilms based on the cluster size and cluster intensity. Results are saved in "Competition_results/" and they are scatter plots that are splitted into 4 regions, based on brightness and cluster size.



## How to train the model yourself
### noise2void resources
- this algorithm is non-supervised and does not need a ground truth for training
- take a look at the [GitHub](https://github.com/juglab/n2v) or [paper](https://arxiv.org/abs/1811.10980)
- take a look at [DigitalSreeni Youtube](https://www.youtube.com/watch?v=71wqPyapFGU&ab_channel=DigitalSreeni)

### RandomForestClassifier
- first make a GT, to do so, take a look at [RFC_Draw_GT.ipynb](scripts/RFC_Draw_GT.ipynb) which needs [napari](https://napari.org/stable/tutorials/fundamentals/installation.html) to be installed
- then train the network with [Train_RFC.py](scripts/Train_RFC.py)
- the model can be saved wherever you want, just make sure to change the path in the [Denoising_and_Binarization.py](scripts/Denoising_and_Binarization.py) script when you want to use your own model.
- there is example trainingsdata available at [example_data/RFC_Trainingsdata/](example_data/RFC_Trainingsdata/)


## FAQ
- **Q:** I get an error when I try to install the package with pip install competitionassay <br>
Try to set up an empty conda environment with python 3.9. Then instead of using the pip install command, download the requirements_no_DL.txt file from the repository and type:
```bash
conda activate competition_env
conda install --file requirements_no_tf_no_napari.txt
```
Then continue running the scripts until they lead to errors like: No module named ... <br>
Install these modules with:
```bash
pip install tensorflow=2.12.0
pip install n2v==0.3.2
pip install napari[all]
```


***
### Author
***
Eric Schmidt