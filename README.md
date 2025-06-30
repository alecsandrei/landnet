# landnet

A repository containing all the code used in my master's thesis, where I'll apply deep learning to detect landslides.

## Conferences

- Cuvuliuc, A.-A., Ursu, D.-E., & Niculiţă, M. (2024, November 16). Using convolutional neural networks to detect landslides from high resolution digital elevation models. National Symposium of Geography Students “Mihai David” 2024, Iași.
- Cuvuliuc, A.-A., Ursu, D.-E., & Niculiţă, M. (2025, April 5). Landslide detection through convolutional neural networks and the PyTorch and Ray libraries. National Symposium of Geography Students, XXXI edition, Bucharest.
- Cuvuliuc, A.-A., Ursu, D.-E., & Niculiță, M. (2025). Evaluating the performance of geomorphometric variables for landslide detection using convolutional neural networks (Nos. EGU25-12939). EGU25. Copernicus Meetings. https://doi.org/10.5194/egusphere-egu25-12939

## Project Organization

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

```
├── LICENSE            <- The license for the project
├── Makefile           <- Makefile with convenience commands like `make data` or `make train` (TODO?)
├── README.md          <- Details about the project
├── data
│   ├── external       <- Data from third party sources (Prut-Bârlad Water Administration LiDAR DEM data)
│   ├── interim        <- Intermediate data that has been transformed (SAGA GIS grids, resampled rasters, etc.)
│   ├── processed      <- The final, canonical data sets resulted from modelling (predicted landslides, metrics, etc.)
│   └── raw            <- The original data (landslide inventory, DEM tile boundaries, etc.)
├── docs               <- TODO
├── landnet            <- Library code for the project
│   ├── modeling           <- Trains, tunes, infers and evaluates models
│   │   ├── classification     <- Handles classification tasks
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py         <- Defines the PyTorch dataset for the classification model
│   │   │   ├── inference.py       <- Runs inference on the classification model
│   │   │   ├── lightning.py       <- PyTorch Lightning module for the classification model
│   │   │   ├── models.py          <- Builds the classification model
│   │   │   ├── stats.py           <- Performs model evaluation
│   │   │   └── train.py           <- Trains the classification model  
│   │   ├── segmentation       <- Handles semantic segmentation tasks
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py         <- Defines the PyTorch dataset for the segmentation model
│   │   │   ├── inference.py       <- Runs inference on the segmentation model
│   │   │   ├── lightning.py       <- PyTorch Lightning module for the classification model
│   │   │   ├── models.py          <- Builds the segmentation model
│   │   │   └── stats.py           <- Performs model evaluation
│   │   ├── __init__.py
│   │   ├── dataset.py         <- Base class for datasets used in the models
│   │   ├── models.py          <- Base class for building models
│   │   └── tune.py            <- Hyperparameter tuning configurations
│   ├── features           <- Feature and dataset creation
│   │   ├── __init__.py
│   │   ├── dataset.py         <- Reading and writing datasets, mostly vector data
│   │   ├── grids.py           <- Creates and manipulates raster datasets
│   │   └── tiles.py           <- Creates and manipulates raster tiles
│   ├── __init__.py
│   ├── config.py          <- Store useful variables and configuration
│   ├── enums.py           <- Enumerations used throughout the project
│   ├── logger.py          <- Loads logging configuration 
│   ├── plots.py           <- Creates visualizations
│   ├── typing.py          <- Type definitions for the project
│   └── utils.py           <- Utility functions used throughout the project
├── logging            <- Logging configurations and files
├── models             <- Trained and serialized models, model predictions and model summaries
├── notebooks          <- Jupyter notebooks
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── maps           <- Generated maps (also the ones made with external software)
├── scripts            <- Python scripts (model inference, feature/dataset creation etc.)
├── .pre-commit-config.yaml
├── pyproject.toml
└── requirements_dev.txt
```
--------

