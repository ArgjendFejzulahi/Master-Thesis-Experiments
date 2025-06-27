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

Here are the commands to run the specific experiments. The numbering is to be understood as the specific section thereby experiment in the masterthesis. The results are all saved in the results folder. There you can find Jupyter Notebooks for visualizaton as well as instructions on how to use them.  Make sure Jupyter Notebooks is installed. You can find the parameters and their description of the specific experiments at the beginning of every script we mention here. The current default parameters are set such that the code can easily run on your local machine. To replicate the experiments you will find the parameter as well in the beginning of the respective script for the EIT experiments. The Experiments for the guillermo dataset have the default settings. 

### Section 4.1 Replicating Reults from Machine Learning Applications: guillermo.py

#### This experiment replicates the reults from the original paper of the nyström preconditioner on the guillermo dataset. The parameter in this script is the number of workers since the script computes the full regularization path; See more detals in Section 4.1. Run the code: 

````
python Guillermo_replication_and_simulation\guillermo.py
````
#### The output is: guillermo_replication_results.xlsx
#### The Notebook for visualization: Analysis_Results_Guillermo.ipynb

### Section 4.2 Condition Number estiomation: condition_number_monte_carlo_parallel.py

### This experiment runs multiple trials in parallel. Constructs a nyström preconditioner, calculates the condition number and saves the results. The script is defined for two regularization parameters mu1, mu2. They are left as their default values like in the thesis. 
Run the code: 

````
python Guillermo_replication_and_simulation\condition_number_monte_carlo_parallel.py
````

#### The output are: guillermo_condition_numbers_mu_0.1.csv, guillermo_condition_numbers_mu_0.01.csv since mu1 = `0.1`, mu2=`0.01`. The name will be adjusted by the choice of parameter. 
#### The Notebook for visualization: Analysis_Condition_Number_Guillermo_MC.ipynb


### Section 4.3 Solving the Inverse Conductivity Problem with Nyström PCG: eit_non_preconditioned.py; eit_preconditioned.py

### These two scripts run the EIT experiment, record the convergence and plot the reconstructed conductivity. They are seperated due to runtime capacities and memory requierments. This enables the user to run them in parallel if more than one compute node is available. The parameters can be found in the beginning of the scripts and also an explanation. Also only one regularizaton parameter at a time can be set up. The experiments were carried out for large dimensions so computational capacity was limited.Run the script as module: 

````
python -m Inverse_EMT.eit_non_preconditioned
````

````
python -m Inverse_EMT.eit_preconditioned
````
#### The outputs are for i.e. mu=1e-7: reconstruction_non_preconditioned_mu_1e-07.png; reconstruction__preconditioned_mu_1e-07.png; EIT_results_cg_mu_1e-07.csv; EIT_results_nyst_pcg_mu_1e-07.csv
#### The notebook for visualization: Analysis_Results_EIT_Experiment.ipynb

### Section 4.4 Optimality of the Preconditioner: eit_non_preconditioned.py; test_P_optimality.py

### This script runs the experiment on the optimal preconditioner per regularization parameter. Given a regularization parameter and a array of mesh_sizes this experiment calculates. the iteration number and time per meshsize. Note: This script containts its own custom functions due to the fact that different requirements for convergence must be met (see Section 4.3 for more details). Therefore this section containts a custom CG-solver and PyEIT solver. 

````
python -m Inverse_EMT.test_P_optimality.py
````
#### The outputs is e.g. for `1e-7`: results_P_optimality_test_mu_1e-07.csv (see in the thesis which ones were tested)
#### The notebook for visualization: Analysis_Optimal_Preconditioner.ipynb






