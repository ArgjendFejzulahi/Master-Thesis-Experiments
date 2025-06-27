# Masterthesis-Experiments

## Overview

This is the GitHub repository from my master’s thesis:  
**"Preconditioning Large Scale Linear Systems: A Randomized Numerical Linear Algebra Approach"**.

It contains the core modules and scripts required to replicate the experiments and results from the thesis.

---

## Setup Instructions

To run the code, follow these steps:

### 1. Navigate to the root directory

### 2. Create and Activate a Virtual Environment (Recommended)



```bash
python -m venv venv
venv\Scripts\activate
````
### 3. Install Python requirements.txt

```bash
pip install -r requirements.txt

````

## Run experiments

Here are the commands to run the specific experiments. The numbering is to be understood as the specific section thereby experiment in the masterthesis. The results are all saved in the results folder. There you can find Jupyter Notebooks for visualizaton as well as instructions on how to use them.  You can find the parameters and their description of the scripts at the beginning of every script we mention here. The current default parameters are set such that the code can easily run on your local machine. To replicate the experiments you will find the parameter as well in the beginning of the respective script for the EIT experiments. The Experiments for the guillermo dataset have the default settings. 

### 4.1 Replicating Reults from Machine Learning Applications: guillermo.py

#### This experiment replicates the reults from the original paper of the nyström preconditioner on the guillermo dataset. The parameter in this script is the number of workers since the script computes the full regularization path; See more detals in Section 4.1. Run the code: 

````
python Guillermo_replication_and_simulation\guillermo.py
````
The output is 

### 4.2








