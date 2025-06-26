import sys
import os

experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(experiments_dir)

import time
import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.sparse import eye
from scipy.sparse.linalg import aslinearoperator, cg
import concurrent.futures

from Nystroem.nystroem import * 


data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'guillermo_data.arff')
data = arff.loadarff(data_path)
print("data succesfully imported")

data = pd.DataFrame(data[0])
y  = np.array(data["class"], dtype = np.float32)
X = data.drop('class', axis=1).to_numpy()
n,d = np.shape(X)
X_sym =  ( 1/n ) * X.T @ X
rhs = (1/n) *  X.T @ y

print("The shape of X_sym")
print(X_sym.shape)
print("The shape of rhs")
print(rhs.shape)

def calc_d_eff(A, mu) -> int:
    dim = A.shape[0]
    return round(np.trace(A @ np.linalg.pinv(A + mu * np.eye(dim))))

# Result storage
effective_dimension = []
rank_mean = []
rank_std = []
cg_time_mean = []
pcg_time_mean = []
cg_time_std = []
pcg_time_std = []

mu_vals = [(10**i) for i in np.linspace(3, -5, 30)]
K_sym_op = aslinearoperator(X_sym)

max_l = round(X_sym.shape[0] * 0.4)

counter = 0
for mu in mu_vals:
    print("This is step: {counter}")
    effective_dimension.append(calc_d_eff(X_sym, mu))
    
    rank = []
    times_cg = []
    times_pcg = []
    
    nys = Nyström(K_sym_op, mu)
    K_mu_op = K_sym_op + aslinearoperator(mu * eye(X_sym.shape[0]))

    def trial_run(_):
        print("Trial number: {_}")
        nys.adaptive_approximation(10, max_l, 35)
        P_nys = nys.preconditioner()
        
        start_cg = time.time()
        _, _ = cg(A = K_mu_op, b = rhs, maxiter=500, rtol = 1e-10)
        end_cg = time.time()
        
        start_pcg = time.time()
        _, _ = cg(A = K_mu_op, b = rhs, M = P_nys, maxiter=500, rtol = 1e-10)
        end_pcg = time.time()
        
        return nys.U.shape[1], end_cg - start_cg, end_pcg - start_pcg

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(trial_run, range(20)))

    # Correct unpacking
    ranks, times_cg_list, times_pcg_list = zip(*results)

    rank.extend(ranks)
    times_cg.extend(times_cg_list)
    times_pcg.extend(times_pcg_list)

    rank_mean.append(np.mean(rank))
    rank_std.append(np.std(rank))
    cg_time_mean.append(np.mean(times_cg))
    pcg_time_mean.append(np.mean(times_pcg))
    cg_time_std.append(np.std(times_cg))
    pcg_time_std.append(np.std(times_pcg))

    counter += 1

# Save results
df = pd.DataFrame({
    'mu_vals': mu_vals,
    'd_eff': effective_dimension,
    'rank_mean': rank_mean,
    'rank_std': rank_std,
    'time_cg_mean': cg_time_mean,
    'time_pcg_mean': pcg_time_mean,
    'time_cg_std': cg_time_std,
    'time_pcg_std': pcg_time_std,
})

df.to_excel("guillermo_replication_results.xlsx", index=False)
