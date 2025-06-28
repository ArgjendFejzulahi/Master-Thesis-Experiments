# Masterthesis-Experiments

## Overview

This is the GitHub repository from my master’s thesis:  
**"Preconditioning Large Scale Linear Systems: A Randomized Numerical Linear Algebra Approach"**.

It contains the core modules and scripts required to replicate the experiments and results from the thesis.

---

## Setup Instructions

To run the code, follow these steps:

### 1. *Navigate to the root directory* (Important) 

### 2. Create and Activate a Virtual Environment (Recommended)

```bash
pyenv install 3.11.2       # Install Python 3.11.2
pyenv local 3.11.2         # Set local Python version to 3.11.2
python -m venv venv        # Create a virtual environment named 'venv'

# Activate the virtual environment
# On Windows:
source venv/Scripts/activate
# On macOS/Linux:
# source venv/bin/activate

````

If this doesn't work you might use conda. It is improtant that the specific python version can be used. Newer versions i.e. python 3.12.X might cause dependencie issues. Note that conda distribution must be downloaded if not available (https://www.anaconda.com/download)

```
conda create -n masterthesis-env python=3.11 -y
conda activate masterthesis-env
```



### 3. Install Python requirements.txt

```bash
pip install -r requirements.txt

````
---

## Run experiments

Here are the commands to run the specific experiments. The numbering corresponds to the specific sections and experiments in the master’s thesis. All results are saved in the `results` folder. There, you can find Jupyter Notebooks for visualization as well as instructions on how to use them. Make sure Jupyter Notebook is installed. Depending on your system you might need to change th directory `results` for the specific visualizations. Caution when choosing regularization parameters since you may overwrite the established results from the thesis.

You can find the parameters and their descriptions for each experiment at the beginning of every script mentioned here. The current default parameters are set so that the code can easily run on your local machine. To replicate the experiments, you will also find the parameters at the beginning of the respective scripts for the EIT experiments. The experiments for the Guillermo dataset use the default settings.


### Section 4.1 Replicating Reults from Machine Learning Applications: `guillermo.py`

This experiment replicates the results from the original paper on the Nyström preconditioner using the Guillermo dataset. The main parameter in this script is the number of workers, since the script computes the full regularization path. See more details in Section 4.1.

Run the code:

````
python Guillermo_replication_and_simulation\guillermo.py
````
The output is:  ```guillermo_replication_results.xlsx ```
The Notebook for visualization:  ```Analysis_Results_Guillermo.ipynb ```

----

### Section 4.2 Condition Number estiomation: `condition_number_monte_carlo_parallel.py`

This experiment runs multiple trials in parallel, constructs a Nyström preconditioner, calculates the condition number, and saves the results. The script uses two regularization parameters, mu1 and mu2, which are left at their default values as in the thesis.
Run the code: 

````
python Guillermo_replication_and_simulation\condition_number_monte_carlo_parallel.py
````

The output are: ```guillermo_condition_numbers_mu_0.1.csv```, ```guillermo_condition_numbers_mu_0.01.csv``` since mu1 = `0.1`, mu2=`0.01`. The name will be adjusted by the choice of parameter. 
The Notebook for visualization:  ```Analysis_Condition_Number_Guillermo_MC.ipynb ```

---

### Section 4.3 Solving the Inverse Conductivity Problem with Nyström PCG: `eit_non_preconditioned.py`; `eit_preconditioned.py`

These two scripts run the EIT experiment, record the convergence, and plot the reconstructed conductivity. They are separated due to runtime and memory requirements. This enables the user to run them in parallel if more than one compute node is available. The parameters and their explanations can be found at the beginning of the scripts. Also, only one regularization parameter can be set at a time. The experiments were carried out for large dimensions, so computational capacity was limited.

Run the script as a module:

````
python -m Inverse_EMT.eit_non_preconditioned
````

````
python -m Inverse_EMT.eit_preconditioned
````
The outputs are for i.e. mu=  ```1e-7``` : ```reconstruction_non_preconditioned_mu_1e-07.png```;  ```reconstruction__preconditioned_mu_1e-07.png ```;  ```EIT_results_cg_mu_1e-07.csv ```;  ```EIT_results_nyst_pcg_mu_1e-07.csv ```
The notebook for visualization:  ```Analysis_Results_EIT_Experiment.ipynb ```

---

### Section 4.4 Optimality of the Preconditioner: `eit_non_preconditioned.py`; `test_P_optimality.py`

This script runs the experiment on the optimal preconditioner for each regularization parameter. Given a regularization parameter and an array of mesh sizes, this experiment calculates the iteration number and time per mesh size. 
Note: This script contains its own custom functions because different convergence requirements must be met (see Section 4.3 for more details). Therefore, this section includes a custom CG solver and a PyEIT solver.


````
python -m Inverse_EMT.test_P_optimality.py
````
The outputs is e.g. for  ```1e-7 ```:  ```results_P_optimality_test_mu_1e-07.csv ``` (see in the thesis which ones were tested)
The notebook for visualization:  ```Analysis_Optimal_Preconditioner.ipynb ```

---

## Modules and Functions

The thesis builds on two Python modules. The Nyström module contains the Nyström approximation algorithm (`Nystroem/nystroem.py`), and the custom Gauss-Newton-Nyström PCG solver is designed for the PyEIT package (`Inverse_EIT/customJac.py`). We would like to quickly introduce the user to their functionality.

### Build a nyström preconditioner: 

```
    np.random.seed(42)
    n = 100;  X = np.random.randn(n, n)
    # ensure symmetry
    A_mat = X @ X.T; mu = 1e-5 #regularization param

    # Define LinearOperator from the matrix
    def matvec(v):
        return A_mat @ v
    A = LinearOperator(shape=(n, n), matvec=matvec, dtype=A_mat.dtype)

    nys = Nyström(A, mu)
    r = 20 # low-rank
    nys.approximation(r)
    P = nys.preconditioner() #get preconditioner

    # Adaptive algorithm
    l0 = 10; lmax = 50; tau = 35
    nys.adaptive_approximation(l0, lmax, tau)
    P_new = nys.preconditioner() #gets overwritten

```

### Custom Gauss-Newton-Nyström PCG (customJac.py)

Extension of pyeit.eit.jac.JAC (see https://github.com/eitcom/pyEIT) : Inherits and extends functionality from PyEIT’s standard Jacobian-based solver. Uses CG method instead of LU. Has the option to construct a randomized nyström preconditioner on the fly and solve the subproblem with Nyström-PCG. The function iside is called gn_custom() (Gauss-Newton-custom) since the authors use gn(). 

````
# .... simulate EIT conductivity (see script eit_preconditioned.py/eit_non_preconditioned.py)
# .... solve Forward model EIT where mesh_obj, protocol_obj are generated
mu = 1e-7 # regularization parameter
if_preconditioner = True # such that nyström preconditioner is used
eit = CustomJAC(mesh_obj, protocol_obj) # set up class
eit.setup(lamb=mu, perm=1, jac_normalized=True)

# You can change nyström approximation parameters within the attributes of the class
eit.start_rank = 200 # start rank of adaptive algorithm
eit.max_rank = 1000 # max rank
eit.tau_param = 35 # tau parameter

#You can set parameters for conjugate gradient i.e.
eit.cg_tol = 1e-10 # detault
eit.cg_max_iter = 350 # default

# run custom Gauss-Newton-Nyström PCG 
s = eit.gn_custom(v=v1, verbose=False, gtol=0.001, maxiter=1, preconditioner=if_preconditioner)

# extract cg info
num_it = eit.cg_iterations
res = eit.residuals




