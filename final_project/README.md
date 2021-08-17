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
    |   ├── aclImdb               [directory containing the dataset files for sentiment polarity classification]
    |   |   ├── test                [test directory of the dataset]
    |   |   |   ├── neg               [test negative directory of the dataset]
    |   |   |   └── pos               [test positive directory of the dataset]
    |   |   └── train               [train directory of the dataset]
    |   |       ├── neg               [train negative directory of the dataset]
    |   |       └── pos               [train positive directory of the dataset]
    |   └── rotten_imdb           [directory containing the dataset files for subjectivity detection]
    |       ├── plot.tok.gt9.5000   [objective sentences file of the dataset]
    |       └── quote.tok.gt9.5000  [subjective sentences file of the dataset]
    ├── main                    [script containing the main code and functions implemented]
    ├── README                  [readme with instructions for running the code]
    ├── report                  [report briefly describing the logic behind the code]
    └── requirements            [requirements of the code]


## Setup

### Required Python Packages

Required Python packages can be installed running the following command:

```bash
    pip3 install -r requirements.txt
```

Once the requirements are installed, the code requires the English models that could be installed running: 

```bash
    python3 -m spacy download en_core_web_sm
```


## Usage

The main script can be run with the following command:

```bash
    python3 main.py
```
