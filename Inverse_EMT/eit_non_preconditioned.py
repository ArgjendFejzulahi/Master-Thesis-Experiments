from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyeit.eit.jac as jac
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
from pyeit.eit.interp2d import sim2pts
from pyeit.mesh.shape import thorax
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from Inverse_EMT.customJac import *
import time
import random


""""--------- Choose parameters : ---------- """
# Parameters used in the thesis ! 
#no_trials = 5 #number of trials to repeat the experiment
#mu = 1e-7 # regularization parameter
#mu_str = f"{mu:.0e}"
#no_electrodes = 24
#mesh_size = 0.01
#if_preconditioner = False

# Those are some toy parameters one might want to use to test the script: 
no_trials = 5 #number of trials to repeat the experiment
mu = 1e-5 # regularization parameter
mu_str = f"{mu:.0e}"
no_electrodes = 16
mesh_size = 0.05
if_preconditioner = False
""" -------------------------"""


print("Running EIT script: WITHOUT preconditioner")
np.random.seed(42)
random.seed(42)
# Build mesh
n_el = no_electrodes
mesh_obj = mesh.create(n_el, h0=mesh_size)
pts, tri = mesh_obj.node, mesh_obj.element
x, y = pts[:, 0], pts[:, 1]

# Define anomalies (conductivities)
anomaly = PyEITAnomaly_Circle(center=[-0.4, -0.2], r=0.2, perm=1000.0)
anomaly2 = PyEITAnomaly_Circle(center=[0.4, 0.2], r=0.2, perm=1000.0)
mesh_new = mesh.set_perm(mesh_obj, anomaly=[anomaly, anomaly2])

# Solve forward problem
protocol_obj = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="std")
fwd = EITForward(mesh_obj, protocol_obj)
v1 = fwd.solve_eit(perm=mesh_new.perm)

nonprec_times, nonprec_residuals, nonprec_cg_iterations = [], [], []
max_len = 0

for t in range(no_trials):
    print(f"Trial {t+1}")
    start_time = time.time()
    eit = CustomJAC(mesh_obj, protocol_obj)
    eit.setup(lamb=mu, perm=1, jac_normalized=True)
    ds = eit.gn_custom(v=v1, verbose=False, gtol=0.001, maxiter=1, preconditioner=if_preconditioner) # run Gauss-Newton without Nyström preconitioner i.e. peconditioner = False
    runtime = time.time() - start_time

    nonprec_times.append(runtime)
    nonprec_residuals.append(np.array(eit.residuals))
    nonprec_cg_iterations.append(np.array(eit.cg_iterations))
    max_len = max(max_len, len(eit.residuals))

# Post-processing
residuals_padded = np.full((no_trials, max_len), np.nan)
for i, res in enumerate(nonprec_residuals):
    residuals_padded[i, :len(res)] = res

results = {
    "mean_cg_it": np.mean(nonprec_cg_iterations),
    "std_cg_it": np.std(nonprec_cg_iterations),
    "mean_residuals": np.mean(residuals_padded, axis=0),
    "std_residuals": np.std(residuals_padded, axis=0, ddof=1),
    "mean_runtime": np.mean(nonprec_times),
    "std_runtime": np.std(nonprec_times, ddof=1),
}

pd.DataFrame([results]).to_csv(f"results/EIT_results_cg_mu_{mu_str}.csv", index=False)

# Plot
ds_last = sim2pts(pts, tri, np.real(ds))
delta_perm = mesh_new.perm - mesh_obj.perm
fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(9, 4))
axes[0].tripcolor(x, y, tri, np.real(delta_perm), shading="flat")
axes[0].set_aspect("equal")
axes[0].set_title("Input Δ Conductivities")
axes[1].tripcolor(x, y, tri, ds_last, shading="flat")
for i, e in enumerate(mesh_obj.el_pos):
    axes[1].annotate(str(i + 1), xy=(x[e], y[e]), color="r")
axes[1].set_aspect("equal")
axes[1].set_title("Reconstructed Δ Conductivities WITHOUT Preconditioner")
fig.colorbar(axes[1].collections[0], ax=axes.ravel().tolist())
plt.savefig(f'results/reconstruction_non_preconditioned_mu_{mu_str}.png', dpi=96)
plt.close()

