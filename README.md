# landnet

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A repository containing all the code used in my master's thesis, where I'll apply deep learning to detect landslides.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train` (TODO?)
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── landnet            <- Source code for use in this project
│   ├── __init__.py
│   │
│   ├── config.py               <- Store useful variables and configuration
│   │
│   ├── dataset.py              <- Scripts to download or generate data
│   │
│   ├── features.py             <- Code to create features for modeling
│   │
│   ├── modeling
│   │   ├── __init__.py
│   │   ├── train.py            <- Code to train models
│   │   └── stats.py            <- Statistical measures for the models
│   │
│   └── plots.py                <- Code to create visualizations
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── figures        <- Generated graphics and figures to be used in reporting
│   └── maps           <- Generated maps (also the ones made with external software)
│
├── scripts            <- Python scripts (model inference, feature/dataset creation etc.)
│
├── .pre-commit-config.yaml     <- Configuration for pre-commit
│
├── pyproject.toml              <- Project configuration file
│
└── requirements_dev.txt        <- The requirements file for contributors
```
--------

