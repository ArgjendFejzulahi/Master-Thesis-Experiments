""""--------- Choose parameters : ---------- """
# Parameters used in the Thesis ! 
# meshsizes = [0.1, 0.09, 0.07, 0.05, 0.03, 0.01, 0.009]
# trials = 5 #number of trials to repeat the experiment
# mu = 1e-7 # regularization parameter (theis used 1e-4, 1e-5 ,1e-6, 1e-7, 1e-8)
# cg_tol = 1e-8 # tolerance for the conjugate gradient
#no_electrodes = 24
#mesh_size = 0.01

# Those are some toy parameters one might want to use to test the script:
meshsizes = [0.1, 0.09, 0.07]
trials = 5 #number of trials to repeat the experiment
mu = 1e-3 
cg_tol = 1e-8 
no_electrodes = 16

""" -------------------------"""

import sys
import os

# Add the experiments folder (parent directory) to sys.path
experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(experiments_dir)
from typing import Union, Optional
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import aslinearoperator, cg
from scipy.sparse import eye, diags
import pandas as pd
#import pyeit.eit.jac as jac
from pyeit.eit.jac import JAC
from Nystroem.nystroem import *

import matplotlib.pyplot as plt
import pyeit.eit.jac as jac
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
from pyeit.eit.interp2d import sim2pts
from pyeit.mesh.shape import thorax
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Circle

import warnings
from scipy.sparse.linalg._interface import LinearOperator
from scipy.linalg import get_lapack_funcs
from numpy import asanyarray, asarray, array, zeros
from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator, \
     IdentityOperator

mu_str = f"{mu:.0e}"

_coerce_rules = {('f','f'):'f', ('f','d'):'d', ('f','F'):'F',
                 ('f','D'):'D', ('d','f'):'d', ('d','d'):'d',
                 ('d','F'):'D', ('d','D'):'D', ('F','f'):'F',
                 ('F','d'):'D', ('F','F'):'F', ('F','D'):'D',
                 ('D','f'):'D', ('D','d'):'D', ('D','F'):'D',
                 ('D','D'):'D'}


def coerce(x,y):
    if x not in 'fdFD':
        x = 'd'
    if y not in 'fdFD':
        y = 'd'
    return _coerce_rules[x,y]


def id(x):
    return x

def make_system(A, M, x0, b):
    A_ = A
    A = aslinearoperator(A)

    if A.shape[0] != A.shape[1]:
        raise ValueError(f'expected square matrix, but got shape={(A.shape,)}')

    N = A.shape[0]

    b = asanyarray(b)

    if not (b.shape == (N,1) or b.shape == (N,)):
        raise ValueError(f'shapes of A {A.shape} and b {b.shape} are '
                         'incompatible')

    if b.dtype.char not in 'fdFD':
        b = b.astype('d')  # upcast non-FP types to double

    def postprocess(x):
        return x

    if hasattr(A,'dtype'):
        xtype = A.dtype.char
    else:
        xtype = A.matvec(b).dtype.char
    xtype = coerce(xtype, b.dtype.char)

    b = asarray(b,dtype=xtype)  # make b the same type as x
    b = b.ravel()

    # process preconditioner
    if M is None:
        if hasattr(A_,'psolve'):
            psolve = A_.psolve
        else:
            psolve = id
        if hasattr(A_,'rpsolve'):
            rpsolve = A_.rpsolve
        else:
            rpsolve = id
        if psolve is id and rpsolve is id:
            M = IdentityOperator(shape=A.shape, dtype=A.dtype)
        else:
            M = LinearOperator(A.shape, matvec=psolve, rmatvec=rpsolve,
                               dtype=A.dtype)
    else:
        M = aslinearoperator(M)
        if A.shape != M.shape:
            raise ValueError('matrix and preconditioner have different shapes')

    # set initial guess
    if x0 is None:
        x = zeros(N, dtype=xtype)
    elif isinstance(x0, str):
        if x0 == 'Mb':  # use nonzero initial guess ``M @ b``
            bCopy = b.copy()
            x = M.matvec(bCopy)
    else:
        x = array(x0, dtype=xtype)
        if not (x.shape == (N, 1) or x.shape == (N,)):
            raise ValueError(f'shapes of A {A.shape} and '
                             f'x0 {x.shape} are incompatible')
        x = x.ravel()

    return A, M, x, b, postprocess


def _get_atol_rtol(name, b_norm, atol=0., rtol=1e-5):
    """
    A helper function to handle tolerance normalization
    """
    if atol == 'legacy' or atol is None or atol < 0:
        msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
               "if set, `atol` must be a real, non-negative number.")
        raise ValueError(msg)

    atol = max(float(atol), float(rtol) * float(b_norm))

    return atol, rtol
    
def custom_cg(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):
    """Use Conjugate Gradient iteration to solve ``Ax = b``.

    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    bnrm2 = np.linalg.norm(b)

    #atol, _ = _get_atol_rtol('cg', bnrm2, atol, rtol)
    atol = 1e-8
    #if bnrm2 == 0:
    #    return postprocess(b), 0

    n = len(b)

    if maxiter is None:
        maxiter = n*10

    dotprod = np.vdot if np.iscomplexobj(x) else np.dot

    matvec = A.matvec
    psolve = M.matvec
    r = b - matvec(x) if x.any() else b.copy()

    # Dummy value to initialize var, silences warnings
    rho_prev, p = None, None

    for iteration in range(maxiter):
        if np.linalg.norm(x, ord = np.inf) < atol:  # Are we done?
            return postprocess(x), 0

        z = psolve(r)
        rho_cur = dotprod(r, z)
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:  # First spin
            p = np.empty_like(r)
            p[:] = z[:]

        q = matvec(p)
        alpha = rho_cur / dotprod(p, q)
        x += alpha*p
        r -= alpha*q
        rho_prev = rho_cur

        if callback:
            callback(x)

    else:  # for loop exhausted
        # Return incomplete progress
        return postprocess(x), maxiter
    
    
class CustomJAC(JAC): 
    """ custom Jac with custom CG for Ax = 0 with infinity norm for optimality experiment"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cg_iterations = []
        self.residuals = [] 
        self.U_shape = None

    def gn_custom(
        self,
        v: np.ndarray,
        x0: Optional[Union[int, float, complex, np.ndarray]] = None,
        maxiter: int = 1,
        gtol: float = 1e-4,
        lamb: Optional[float] = None,
        lamb_decay: float = 1.0,
        lamb_min: float = 0.0,
        preconditioner: bool = False,
        verbose: bool = True,
        generator: bool = False,
        **kwargs,
    ):
        self._check_solver_is_ready()
        if x0 is None:
            x0 = self.mesh.perm_array
        if lamb is None:
            lamb = self.params["lamb"]
        
        x0_norm = np.linalg.norm(x0)

        def generator_gn():
            nonlocal x0, lamb
            for i in range(maxiter):
                jac, v0 = self.fwd.compute_jac(x0)
                r0 = v - v0
                
                jac_lin_op = aslinearoperator(jac)
                j_w_j = jac_lin_op.T @ jac_lin_op

                #j_w_j = aslinearoperator(jac.T @ jac)
                reg = aslinearoperator(lamb * eye(jac.shape[1]))
                A_lam  = j_w_j + reg
                
                #b = jac.T @ r0
                b = np.zeros(j_w_j.shape[0])
                
                def callback_counter(xk):
                    callback_counter.iterations += 1
                    #self.residuals.append(np.linalg.norm(b - A_lam.matvec(xk)))
                    self.residuals.append(np.linalg.norm(xk, np.inf))
                    #self.residuals.append(np.linalg.norm(b - A_lam.matvec(xk), ord = np.inf))
                    
                callback_counter.iterations = 0
                m = b.shape; 
                #self.residuals.append(np.linalg.norm(b - A_lam.matvec(np.ones(m)*x0)))
                #self.residuals.append(np.linalg.norm(np.ones(m)*x0, np.inf))
                self.residuals.append(np.linalg.norm(x0, ord = np.inf))
                
                #print("Norm of x0 :")
                #print(np.linalg.norm(x0, ord = np.inf))                
                if preconditioner == "nyström":
                    nys = Nyström(j_w_j, lamb)
                    nys.adaptive_approximation(100, 4000, 35)
                    P = nys.preconditioner()
                    d_k, cg_info = custom_cg(A_lam, b, x0 = x0, M = P, callback = callback_counter, rtol = cg_tol) 
                    self.U_shape = nys.U.shape
                    
                else: 
                    j_w_j = jac.T @ jac 
                    d_k, cg_info = custom_cg(A_lam, b, callback = callback_counter, rtol = cg_tol)
                
                self.cg_iterations.append(callback_counter.iterations)  
                
                x0 = x0 - d_k 

                c = np.linalg.norm(d_k) / x0_norm
                self.error = c
                if c < gtol:
                    break

                if verbose:
                    print("iter = %d, lamb = %f, gtol = %f" % (i, lamb, c))

                lamb *= lamb_decay
                lamb = max(lamb, lamb_min)
                yield x0

        real_gen = generator_gn
        if not generator:
            item = None
            for item in real_gen():
                pass
            return item
        else:
            return real_gen()


max_len = 0
results = []

for h in meshsizes:
    
    cg_iterations = []
    approx_ranks = []
    residuals = []
    

        
    """ 0. build mesh """
    n_el = no_electrodes
    mesh_obj = mesh.create(n_el, h0=h)

    # extract node, element, alpha
    pts = mesh_obj.node
    tri = mesh_obj.element
    x, y = pts[:, 0], pts[:, 1]

    """ 1. problem setup """
    anomaly = PyEITAnomaly_Circle(center=[0.25, 0.25], r=0.15, perm=1000.0)
    anomaly2 = PyEITAnomaly_Circle(center=[-0.25, -0.25], r=0.15, perm=1000.0)
    mesh_new = mesh.set_perm(mesh_obj, anomaly=[anomaly, anomaly2])

    """ 2. FEM simulation """
    protocol_obj = protocol.create(n_el, dist_exc=8, step_meas=1, parser_meas="std")

    fwd = EITForward(mesh_obj, protocol_obj)
    v0 = fwd.solve_eit()
    v1 = fwd.solve_eit(perm=mesh_new.perm)
        
    for t in range(trials): 
        
        eit = CustomJAC(mesh_obj, protocol_obj)
        eit.setup(lamb=0.00001, perm=1, jac_normalized=True)
        ds = eit.gn_custom(v = v1, gtol = 0.001, maxiter = 1, preconditioner = "nyström")
        
        print("Meshsize: " + str(h))
        print("This is trial: " + str(t))
        print("The Dimension n of J.T @ J :" + str(eit.U_shape[0]))
        print("The Chosen rank by Nyström is : " + str(eit.U_shape[1]))
        print("----------------------------------------")
        
        res = eit.residuals
        residuals.append(np.array(res, dtype=np.float64))
        cg_iterations.append(np.array(eit.cg_iterations, dtype=np.float64))
        approx_ranks.append(np.array(eit.U_shape[1], dtype=np.float64))

        
        if len(res) > max_len:
            max_len = len(res)
    
    
    cg_iterations = np.array(cg_iterations)
    approx_ranks = np.array(approx_ranks)

    #residuals_padded = np.full((len(residuals), max_len), np.nan)
    residuals_padded = np.ones((len(residuals), max_len), np.float64)

    for i, res in enumerate(residuals):
        residuals_padded[i, :len(res)] = res
        

    # csv is updated iteratively ! 
    results_local = {
        "mesh_size": h,
        "matrix_dim": eit.U_shape[0],
        "mean_cg_it": np.mean(cg_iterations),
        "std_cg_it": np.std(cg_iterations),
        "mean_residuals": np.nanmean(residuals_padded, axis=0),
        "std_residuals": np.nanstd(residuals_padded, axis=0, ddof=1),
        "mean_approx_rank": np.mean(approx_ranks),
        "std_approx_rank": np.std(approx_ranks, axis = 0, ddof=1),
        }

    results.append(results_local)

    results_df = pd.DataFrame(results)
    results_df.to_csv(f"results/results_P_optimality_test_mu_{mu_str}.csv")

    
    
