
<br></br>
<img align="right" src="https://www.cmes.info/img/logos/ai4er_logo_2048px.png" width="100" height="100">
<img align="centre" src="https://www.bas.ac.uk/wp-content/uploads/2016/11/BAS_colour-736x164.jpg" width="450" height="100">
<img align="left" src="https://www.cam.ac.uk/sites/www.cam.ac.uk/files/inner-images/logo.jpg" width="400" height="100">
<br><br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# AI4ER Guided Team Challenge 2023: Sea Ice Classification

This repository contains code for the Sea Ice Classification Challenge from the 2022-23 cohort of [AI4ER](https://ai4er-cdt.esc.cam.ac.uk). 
<br><br>
AI4ER is the the UKRI Centre for Doctoral Training (CDT) in the Application of Artificial Intelligence to the study of Environmental Risks at the [University of Cambridge](https://www.cam.ac.uk).

## Project Description

This goal of this 3-month project is to automatically classify sea ice concentration in the East Weddell Sea, Antarctica. The Weddell Sea is an active area of iceberg calving [[1]](https://www.bas.ac.uk/media-post/brunt-ice-shelf-in-antarctica-calves-giant-iceberg/) and a critical shipping route for access to the Halley Research Station, a research facility operated by the British Antarctic Survey. Accurate assessments of sea ice concenration in the East Weddell Sea are hence of great importance to ensure the safety and success of future expeditions.

## Demonstration

TO DO:
- Include link to video
- Include images of before/after human vs. model ice chart maps
- Look into [Hugging Face platform](https://huggingface.co/) platform?

## Data
This project uses two publicly available datasets:
- Labeled sea ice charts provided by the [Norwegian Ice Service](http://ice.aari.aq/antice/)
- Sentinel 1 Synthetic Aperture Radar (SAR) satellite imagery provided by the [European Space Agency](https://www.esa.int) and [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/sentinel)

These datasets are shown below superimposed over the region of interest on Google Earth.

<img width="764" alt="data" src="https://user-images.githubusercontent.com/114443493/224169683-72f51105-c709-43b5-86f5-54f95e49a74e.png">

## Models
This project uses three models:
1. A baseline Random Forest model
2. A basic Unet
3. A pretrained resnet34 from the [segmentation_models_pytorch](https://segmentation-modelspytorch.readthedocs.io/en/latest/) Python library, which is distributed under the MIT license. [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Code Structure
TO DO:
```
├───Data                              <-- containing the satellite images and ice chart data used for this project
│   ├───dual_band_images
│   ├───rasterised_ice_charts
├───Notebooks                         <-- notebooks to demonstrate and recreate our exploratory data analysis, preprocessing, modelling and evaluation
│   ├───example.ipynb
├───Prepocessing                      <-- preprocessing python scripts and data evaluation
|   ├───constants.py
|   ├───info.md
|   ├───interesting_images.csv
|   ├───metrics.csv
|   ├───metrics.py
|   ├───metrics_per_pair.csv
|   ├───tiling.py
├───Model                             <-- modelling python scripts, including Random Forest and Unet
│   ├───model.py
│   ├───split.py
│   ├───test.py
│   ├───train.py
│   ├───train_scikit.py
│   ├───util.py
├───Results                           <-- model evaluation python scripts
│   ├───TBC.py
```

## Workflow

TO DO: Include diagram of preprocessing and modelling steps.

## Usage
An archived copy of this repository at the time of project submission (17th March 2023) is available on Zenodo **(TBC)**. To test the code and recreate the results of this project, follow the steps below: 
1. Clone this repository (for the latest version) or retrieve the archived copy from Zenodo
2. Create and activate the conda environment using ```conda activate environment.yml```, which contains all required Python modules and versions.
3. To generate ice chart and SAR tile pairs of 256x256 dimensions run: ```python tiling.py```. Tile pairs containing NaN values will be discarded.
4. To train the model run: ```python train.py```. Input arguments include:

    | Argument                   | Options          | Default|
    | -------------------------- |:----------------:| ------:|
    | --model                    | unet, resnet34   | unet   |
    | --classification_type      | binary, ternary  | binary |
    | --criterion                | ce, dice, focal  | ce     |
    | --sar_band3                | angle, ratio     | angle  |
  
5. TO DO: To load the model from a checkpoint file run: ```TBC```


## Contributors
Project core members contributed equally to this work:
- [Joshua Dimasaka](https://ai4er-cdt.esc.cam.ac.uk/StaffDirectory/students-all/2022-students), AI4ER MRes Student (2022 Cohort), University of Cambridge
- [Andrew McDonald](https://ai4er-cdt.esc.cam.ac.uk/StaffDirectory/students-all/2022-students), AI4ER MRes Student (2022 Cohort), Univeristy of Cambridge
- [Meghan Plumridge](https://ai4er-cdt.esc.cam.ac.uk/StaffDirectory/students-all/2022-students), AI4ER MRes Student (2022 Cohort), University of Cambridge
- [Jay Torry](https://ai4er-cdt.esc.cam.ac.uk/StaffDirectory/students-all/2022-students), AI4ER MRes Student (2022 Cohort), University of Cambridge
- [Andrés Camilo Zúñiga González](https://ai4er-cdt.esc.cam.ac.uk/StaffDirectory/students-all/2022-students), AI4ER MRes Student (2022 Cohort), University of Cambridge

With special thanks to our advisors for their project guidance and technical support:
- Madeline Lisaus, AI4ER PhD Student (2021 Cohort), University of Cambridge
- Jonathon Roberts, AI4ER PhD Student (2021 Cohort), University of Cambridge
- Martin Rogers, AI-lab, British Antarctic Survey

## References
[1] Brunt ice shelf in Antarctica calves giant iceberg (2023) British Antarctic Survey. Available at: https://www.bas.ac.uk/media-post/brunt-ice-shelf-in-antarctica-calves-giant-iceberg/ (Accessed: March 9, 2023). 
