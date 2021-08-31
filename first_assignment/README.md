# First Assignment

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Structure](#structure)
- [Setup](#setup)
  - [Required Python Packages](#required-python-packages)
- [Usage](#usage)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Structure

    first_assignment
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

The main script to test the functions and see the results of the transition-based dependency parser can be run with the following command:

```bash
    python3 main.py
```

Before run the code, you can also change the sentence to analyze by modifying:

```python
    example_sentence: str = "I saw a man with a telescope, he was looking at the Moon."
```

You can also change the example and wrong span by modifying:

```python
    example_span: Span = spacy_nlp(example_sentence)[2:7]
    wrong_span: Span = spacy_nlp(example_sentence)[5:8]
```
