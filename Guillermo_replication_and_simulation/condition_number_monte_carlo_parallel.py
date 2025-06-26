import sys
import os

experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(experiments_dir)

flag = False
from Nystroem.nystroem import *

from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from scipy.io import arff
import time


data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'guillermo_data.arff')


data = arff.loadarff(data_path)
print("data succesfully imported")
data = pd.DataFrame(data[0])
X = data.drop('class', axis=1).to_numpy()
n,d = np.shape(X)
X_sym =  ( 1/n ) * X.T @ X
f = (1/n) *  X.T @ y

# Mock data example for testing
#X = np.random.randn(100, 100)
#X_sym = X.T @ X
#mu_vals = [(10**i) for i in np.linspace(3, -5, 20)]

start = time.time()
K_sym_op = aslinearoperator(X_sym)
max_l = round(X_sym.shape[0] * 0.4)
max_trials = 200

mu1 = 0.1
mu2 = 0.01

# Function to estimate condition numbers
def estimate_condition_numbers(mu, max_trials):

    K_mu_op = K_sym_op + aslinearoperator(mu*eye(X_sym.shape[0]))

    nystr = Nyström(K_sym_op, mu)
    conds = []
    trials = []

    def compute_trial(t):
        nystr.adaptive_approximation(10, max_l, 35)
        P = nystr.preconditioner()
        cond = cond_2_sparse(P @ K_mu_op)
        return t, cond

    with ThreadPoolExecutor(max_workers = 3) as executor:
        results = list(executor.map(compute_trial, range(max_trials + 1)))

    for t, cond in results:
        trials.append(t)
        conds.append(cond)

    return trials, conds

# Parallel execution for mu1
print("Estimating condition numbers for mu1...")
trials_mu1, mu1_conds = estimate_condition_numbers(mu1, max_trials)
mu1_data = {
    'Trial': trials_mu1,
    'Condition_Number': mu1_conds
}

mu1_df = pd.DataFrame(mu1_data)
mu1_filename = f"results/guillermo_condition_numbers_mu_{mu1}.csv"
mu1_df.to_csv(mu1_filename, index=False)
print(f"Saved mu1 results to {mu1_filename}")

# Parallel execution for mu2
print("Estimating condition numbers for mu2...")
trials_mu2, mu2_conds = estimate_condition_numbers(mu2, max_trials)
mu2_data = {
    'Trial': trials_mu2,
    'Condition_Number': mu2_conds
}
mu2_df = pd.DataFrame(mu2_data)
mu2_filename = f"results/guillermo_condition_numbers_mu_{mu2}.csv"
mu2_df.to_csv(mu2_filename, index=False)
print(f"Saved mu2 results to {mu2_filename}")

end = time.time()

print("time for the computation: " + str(end - start))
