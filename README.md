![Udacity Logo](./docs/logo/udacity-logo.svg)

# AWS Machine Learning Engineer Nanodegree 
## Capstone Project <br/> Arvato Customer Acquisition Prediction Using Supervised Learning

> ### Fady Morris Milad Ebeid  <br/><br/> January 22, 2022


# Documentation
<!-- For the full documentation, refer to [project report](./docs/report.pdf). -->

# Libraries Used
- [Python v3.8.12](https://www.python.org)
- [NumPy v1.22.1](https://numpy.org/)
- [Pandas v1.4.1](https://pandas.pydata.org/)
- [Matplotlib v3.5.1](https://matplotlib.org/)
- [Seaborn v0.11.2](https://seaborn.pydata.org/)
- [scikit-learn v1.0.2](https://scikit-learn.org/stable/)
- [sagemaker v2.72.1](https://sagemaker.readthedocs.io/)
- [boto3 v1.20.26](https://boto3.amazonaws.com/)


# Directory Structure

  
- [notebooks/](./notebooks/) : Contains the project Jupyter notebooks and exported HTML files of the project notebooks.  
- [src/](./src/) : Contains python helper scripts.
- [docs/](./docs/) : Contains the project [report](./docs/report.pdf) and [proposal](./docs/proposal.pdf).  
   
   
**Full project directory structure :**

```
<project root directory>
├── README.md                 - Project readme file.
├── docs/                     - Project documentation directory.
│   ├── proposal.pdf          - Project proposal.
│   └── report.pdf            - Project report.
├── input
│   └── data
│       ├── processed                                            - Project cleaned dataset
│       │   ├── metadata.csv
│       │   ├── test.csv
│       │   ├── train.csv
│       │   └── valid.csv
│       └── raw                                                  - Project raw dataset
│           ├── DIAS Attributes - Values 2017.xlsx                     - Metadata excel file.
│           ├── DIAS Information Levels - Attributes 2017.xlsx
│           ├── Udacity_MAILOUT_052018_TEST.tar.xz                     - Raw testing dataset.
│           └── Udacity_MAILOUT_052018_TRAIN.tar.xz                    - Raw training dataset
├── notebooks                          - Jupyter notebooks
│   ├── 00_common.ipynb                     - Common code that is imported in other notebooks.
│   ├── 01_data-exploration.ipynb           - Exploratory data analysis notebook.
│   ├── 02_data-processing-pipeline.ipynb   - Data processing pipeline notebook.
│   └── 03_classification-model.ipynb       - Model training, tuning and evaluation notebook.
├── output/                                 - Project documentation directory.
│   ├── figures/                            - Project output plots and graphs.
│   ├── submissions                         - Predictions to be submitted to kaggle. 
│   └── tables                              - Project output latex tables and statistics.
└── src                                - Packaged python source code.
    ├── helper_functions.py                 - Helper functions.
    ├── metadata.py                         - Metadata class.
    └── transformers.py                     - Scikit-Learn based transformers.
```


<!--
# Project Directory and Data Preparation

- Clone the repository :

        git clone 
-->

