# Second Assignment

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Structure](#structure)
- [Setup](#setup)
  - [Required Python Packages](#required-python-packages)
- [Usage](#usage)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Structure

    second_assignment
    ├── data                    [data directory]
    |   └── conll2003             [directory containing the dataset files]
    |       ├── dev                 [dev file of the dataset]
    |       ├── test                [test file of the dataset]
    |       └── train               [train file of the dataset]
    ├── conll                   [library to handle the operations on the conll2003 dataset]
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

Once the requirements are installed, the code requires the English model that could be installed running: 

```bash
    python3 -m spacy download en_core_web_sm
```


## Usage

The main script can be run with the following command:

```bash
    python3 main.py
```
