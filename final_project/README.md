# Final Project

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Structure](#structure)
- [Setup](#setup)
  - [Required Python Packages](#required-python-packages)
- [Usage](#usage)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Structure

    final_project
    ├── data                    [data directory]
    |   ├── aclImdb               [directory containing the dataset files for sentiment polarity classification and objectivity detection]
    |   |   ├── test                [test directory of the dataset]
    |   |   |   ├── neg               [test negative directory of the dataset]
    |   |   |   └── pos               [test positive directory of the dataset]
    |   |   └── train               [train directory of the dataset]
    |   |       ├── neg               [train negative directory of the dataset]
    |   |       └── pos               [train positive directory of the dataset]
    |   └── rotten_imdb           [directory containing the dataset files for objectivity detection]
    |       ├── plot.tok.gt9.5000   [objective sentences file of the dataset]
    |       └── quote.tok.gt9.5000  [subjective sentences file of the dataset]
    ├── main                    [notebook containing the main code for the project]
    ├── README                  [readme with instructions for running the code]
    ├── report                  [report describing the work done and the logic behind the code]
    └── requirements            [requirements of the code]


## Setup

### Required Python Packages

Required Python packages can be installed running the following command:

```bash
    pip3 install -r requirements.txt
```

All the required packages are already installed on [Google Colab](https://colab.research.google.com/notebooks/).

The first cell of the notebook forces the installation of the current latest version of [spaCy](https://spacy.io/usage/v3-1) and the installation of the required English model.


## Usage

The main notebook can be executed in [Google Colab](https://colab.research.google.com/notebooks/) loading the notebook or on your local pc using Jupyter and run all the cells.

It is recommended the use of a GPU that supports CUDA framework.
