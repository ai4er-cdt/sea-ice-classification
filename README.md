
<br></br>
<img align="right" src="https://www.cmes.info/img/logos/ai4er_logo_2048px.png" width="75" height="75">
<img align="centre" src="https://www.bas.ac.uk/wp-content/uploads/2016/11/BAS_colour-736x164.jpg" width="350" height="75">
<img align="left" src="https://www.cam.ac.uk/sites/www.cam.ac.uk/files/inner-images/logo.jpg" width="300" height="75">
<br><br>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# AI4ER Guided Team Challenge 2023: <br /> Sea Ice Classification

This repository contains code for the Sea Ice Classification Challenge from the 2022-23 cohort of [AI4ER](https://ai4er-cdt.esc.cam.ac.uk). 
<br><br>
AI4ER is the the UKRI Centre for Doctoral Training (CDT) in the Application of Artificial Intelligence to the study of Environmental Risks at the [University of Cambridge](https://www.cam.ac.uk).

## Project Description

This goal of this 3-month project was to automatically classify sea ice concentration in the East Weddell Sea, Antarctica. The Weddell Sea is an active area of iceberg calving [[1]](https://www.bas.ac.uk/media-post/brunt-ice-shelf-in-antarctica-calves-giant-iceberg/) and a critical shipping route for access to the Halley Research Station, a research facility operated by the British Antarctic Survey. Accurate assessments of sea ice concenration in the East Weddell Sea are hence of great importance to ensure the safety and success of future expeditions.


## Data
This project uses two publicly available datasets:
- Labeled sea ice charts jointly developed by the [Arctic and Antarctic Research Institute, USA National/Naval Ice Center, and Norwegian Meteorological Institute](http://ice.aari.aq/antice/).
- Sentinel 1A and 1B Synthetic Aperture Radar (SAR) satellite imagery provided by the [Copernicus Open Access Hub](https://sentinels.copernicus.eu/web/sentinel/missions/sentinel-1/data-products), operated by the European Space Agency. The Copernicus Open Data Policy enables free, open access to Sentinel products [[2](https://sentinel.esa.int/web/sentinel/faq)]. Sentinel Terms and Conditions can be found at the following [link](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice).


### Region of Interest
<p align="center">
    <img src="https://user-images.githubusercontent.com/114443493/224169683-72f51105-c709-43b5-86f5-54f95e49a74e.png" width="100%"\>
</p>

### Relative locations and areal extents of (b) training, (c) validating, and (d) testing subsets.
<p align="center">
    <img src="readme_images/train_val_test_split.png" width="100%"\>
</p>


### Temporal and geographical coverage of the dataset.
|   Pair      Number   |   Subset   | Date of Sea   Ice      Concentration Chart | Observation   Window      of SAR signals | Geographical      Coverage |
|:--------------------:|:----------:|:------------------------------------------:|:----------------------------------------:|:--------------------------:|
|                 1    |  training  |                   Nov 6, 2017              |        10:25:50 PM -   10:26:55 PM       |                Weddell Sea          |
|                 2    |  training  |                Dec 23, 2017                |        10:42:00 PM -   10:43:04 PM       |                Weddell Sea          |
|                 3    |  training  |                 Jan 4, 2018                |        10:41:59 PM -   10:43:03 PM       |                Weddell Sea          |
|                 4    |  training  |                Feb 22, 2018                |        11:22:34 PM -   11:23:38 PM       |                Weddell Sea          |
|                 5    |  training  |                Feb 23, 2018                |        12:59:54 AM - 1:00:58   AM        |                Weddell Sea          |
|                 6    |  training  |                Feb 26, 2018                |         1:24:13 AM - 1:25:17   AM        |                Weddell Sea          |
|               7      |  training  |                 Dec 3, 2018                |         8:18:15 AM - 8:19:14   AM        |                Antarctica Peninsula          |
|                 8    |  training  |                 Dec 9, 2018                |         8:51:00 AM - 8:52:08   AM        |                Antarctica Peninsula          |
|                 9    |  training  |                Dec 10, 2018                |        11:50:06 PM -   11:51:05 PM       |                Antarctica Peninsula          |
|                10    |  training  |                Dec 20, 2018                |         8:19:15 AM - 8:20:19   AM        |                Antarctica Peninsula          |
|                11    |  training  |                 Mar 5, 2020                |        12:35:47 AM -   12:36:52 AM       |                Weddell Sea          |
|                12    |  training  |                Mar 13, 2020                |         1:08:15 AM - 1:09:20   AM        |                Weddell Sea          |
|                13    |  training  |                Dec 23, 2021                |        11:14:52 PM -   11:15:43 PM       |             Weddell Sea             |
|                14    | validating |                Mar 13, 2019                |        11:22:41 PM -   11:23:45 PM       |                Weddell Sea          |
|                15    | validating |                Jan 17, 2020                |        10:01:39 PM -   10:02:43 PM       |                Weddell Sea          |
|               16     |   testing  |                Dec 16, 2022                |        11:23:48 PM -   11:24:53 PM       |             Weddell Sea             |
|                17    |   testing  |                Dec 22, 2022                |         1:09:23 AM - 1:10:27   AM        |             Weddell Sea             |
|                18    |   testing  |                Jan 12, 2023                |         8:32:41 PM - 8:33:45   PM        |             Weddell Sea             |


### Sample visualizations of (a) water and ice, (b) HH band, (c) HV band, (d) angle, and (e) HH/HV.
<p align="center">
    <img src="readme_images/third_band.png" width="100%"\>
</p>

### Categories of sea ice concentrations.
| **Code** | **Description** | **Code** | **Description** | **Code** | **Description** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | Ice Free | 35 | 3/10 to 5/10 ice | 78 | 7/10 to 8/10 ice |
| 1 | Open Water (	extless 1/10   ice) | 40 | 4/10 ice | 79 | 7/10 to 9/10 ice |
| 2 | Bergy Water | 45 | 4/10 to 5/10 ice | 80 | 8/10 ice |
| 10 | 1/10 ice | 46 | 4/10 to 6/10 ice | 81 | 8/10 to 10/10 |
| 12 | 1/10 to 2/10 ice | 47 | 4/10 to 7/10 ice | 89 | 8/10 to 9/10 ice |
| 13 | 1/10 to 3/10 ice | 50 | 5/10 ice | 90 | 9/10 ice |
| 14 | 1/10 to 4/10 ice | 56 | 5/10 to 6/10 ice | 91 | 9/10 to 10/10 ice,   9+/10 ice |
| 20 | 2/10 ice | 57 | 5/10 to 7/10 ice | 92 | 10/10 ice |
| 23 | 2/10 to 3/10 ice | 60 | 6/10 ice | 99 | Unknown/Undetermined |
| 24 | 2/10 to 4/10 ice | 67 | 6/10 to 7/10 ice | -9 | Null Value |
| 30 | 3/10 ice | 68 | 6/10 to 8/10 ice | 100 | Full Ice |
| 34 | 3/10 to 4/10 ice | 70 | 7/10 ice |  |  |

### Schematic illustration of the overlapping tiling technique.
<p align="center">
    <img src="readme_images/tiling.png" width="50%"\>
</p>

## Models
This project uses three models:
1. A baseline Decision Tree (DT) model
2. A basic U-Net
3. A pretrained resnet34 from the [segmentation_models_pytorch](https://segmentation-modelspytorch.readthedocs.io/en/latest/) Python library, which is distributed under the MIT license. [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Code Structure

```
|──────tutorial                     <-- Folder containing tutorial CNN notebook
|──────info.md                      <-- Sea ice concentration values
|──────JASMIN.md                    <-- Step-by-step guide
|──────constants.py                 <-- SAR/ice chart pairs, binary/ternary classes
|──────environment.yml              <-- List of Python modules
|──────interesting_images.csv       <-- List of tiles containing all three categories for ternary classification
|──────interesting_images.py        <-- Generate interesting_images.csv
|──────intro_deep_learning.ipynb    <-- Deep learning notebook
|──────metrics.csv                  <-- Lists mean and std dev of all SAR images
|──────metrics.py                   <-- Calculate metrics.csv
|──────metrics_per_pair.csv         <-- Lists mean and std dev for individual SAR image
│──────model.py                     <-- Unet model and evaluation metrics
│──────split.py                     <-- Construct training & validation datasets
│──────test.py                      <-- Test CNN and save output to WANDB
|──────test_scikit.py               <-- Test DT and save output to WANDB
|──────test_slurm_script.py         <-- Test the model on JASMIN
|──────test_slurm_script_scikit.py  <-- Test the DT on JASMIN
|──────tiling.py                    <-- Generate tiles from SAR/ice chart paris
│──────train.py                     <-- Train CNN and save output to WANDB
|──────train_scikit.py              <-- Train DT and save output to WANDB
|──────train_slurm_script.sh        <-- Train the model on JASMIN
|──────train_slurm_script_scikit.sh <-- Train the DT on JASMIN
│──────util.py                      <-- Load data into the model and normalise
|──────util_scikit.py               <-- Load data into the model, normalise and create the training dataset
```

## Workflow

### CNN Workflow (U-Net and ResNet34)
<p align="center">
    <img src="readme_images/workflow1.png" width="100%"\>
</p>

### Decision Tree Workflow
<p align="center">
    <img src="readme_images/workflow2.png" width="100%"\>
</p>

## Usage
To test the code and recreate the results of this project, follow the steps below: 
1. Clone this repository (for the latest version) or retrieve the archived copy from Zenodo
2. Create and activate the conda environment using ```conda activate environment.yml```, which contains all required Python modules and versions.
3. To generate ice chart and SAR tile pairs of 256x256 dimensions run: ```python tiling.py```. Tile pairs containing NaN values will be discarded.
4. Follow the steps in [JASMIN.md](https://github.com/ai4er-cdt/sea-ice-classification/blob/dev/JASMIN.md) to train and test the CNN or DT model. Input arguments that were modified for this project include:

    | Argument                   | Options          | Default|
    | -------------------------- |:----------------:| ------:|
    | --model                    | unet, resnet34   | unet   |
    | --classification_type      | binary, ternary  | binary |
    | --sar_band3                | angle, ratio     | angle  |
  

## Contributors
Project core members contributed equally to this work:
- [Joshua Dimasaka](https://ai4er-cdt.esc.cam.ac.uk/StaffDirectory/students-all/2022-students), AI4ER MRes Student (2022 Cohort), University of Cambridge
- [Andrew McDonald](https://ai4er-cdt.esc.cam.ac.uk/StaffDirectory/students-all/2022-students), AI4ER MRes Student (2022 Cohort), Univeristy of Cambridge
- [Meghan Plumridge](https://ai4er-cdt.esc.cam.ac.uk/StaffDirectory/students-all/2022-students), AI4ER MRes Student (2022 Cohort), University of Cambridge
- [Jay Torry](https://ai4er-cdt.esc.cam.ac.uk/StaffDirectory/students-all/2022-students), AI4ER MRes Student (2022 Cohort), University of Cambridge
- [Andrés Camilo Zúñiga González](https://ai4er-cdt.esc.cam.ac.uk/StaffDirectory/students-all/2022-students), AI4ER MRes Student (2022 Cohort), University of Cambridge

With special thanks to our advisors for their project guidance and technical support:
- Madeline Lisaius, AI4ER PhD Student (2021 Cohort), University of Cambridge
- Jonathan Roberts, AI4ER PhD Student (2021 Cohort), University of Cambridge
- Martin Rogers, AI-lab, British Antarctic Survey

## References
[1] Brunt ice shelf in Antarctica calves giant iceberg (2023) British Antarctic Survey. Available at: https://www.bas.ac.uk/media-post/brunt-ice-shelf-in-antarctica-calves-giant-iceberg/ (Accessed: March 9, 2023). 

[2] The European Space Agency (n.d.) FAQ content, FAQ - Sentinel Online - Sentinel Online. Available at: https://sentinel.esa.int/web/sentinel/faq (Accessed: March 17, 2023). 

Significant parts of our deep learning code are inspired by / adapted from the following:

https://github.com/ampersandmcd/DeepExtremeMixtureModel

https://github.com/ampersandmcd/COMETFlows

https://github.com/ampersandmcd/CSE803FinalProject

