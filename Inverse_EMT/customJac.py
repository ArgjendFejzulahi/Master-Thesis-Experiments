import sys
import os
import time

# Add the experiments folder (parent directory) to sys.path
experiments_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(experiments_dir)

from typing import Union, Optional
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import aslinearoperator, cg
from scipy.sparse import eye
#import pyeit.eit.jac as jac
from pyeit.eit.jac import JAC
from Nystroem.nystroem import *


class CustomJAC(JAC): 
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # params for the nyström preconditioner
        self.start_rank = 100
        self.max_rank = 4000
        self.tau_param = 35
        
        self.cg_iterations = []
        self.residuals = [] 
        self.U_shape = None
        self.time_preconditioner = None

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
            x0 = self.mesh.perm
        if lamb is None:
            lamb = self.params["lamb"]

        x0_norm = np.linalg.norm(x0)

        def generator_gn():
            nonlocal x0, lamb
            for i in range(maxiter):
                jac, v0 = self.fwd.compute_jac(x0)
                r0 = v - v0

                j_w_j = aslinearoperator(jac.T @ jac)
                reg = aslinearoperator(lamb * eye(jac.shape[1]))
                A_lam  = j_w_j + reg
                
                b = jac.T @ r0
                
                def callback_counter(xk):
                    callback_counter.iterations += 1
                    self.residuals.append(np.linalg.norm(b - A_lam.matvec(xk)))
                    
                callback_counter.iterations = 0
                m = b.shape; 
                self.residuals.append(np.linalg.norm(b - A_lam.matvec(np.ones(m)*x0)))
                
                if preconditioner:
                    nys = Nyström(j_w_j, lamb)
                    start = time.time()
                    nys.adaptive_approximation(self.start_rank,  self.max_rank, self.tau_param)
                    self.time_preconditioner = time.time() - start
                    
                    P = nys.preconditioner()
                    d_k, cg_info = cg(A_lam, b, M = P, maxiter = 350 ,callback = callback_counter, rtol = 1e-10) 
                    self.U_shape = nys.U.shape
                else: 
                    j_w_j = jac.T @ jac 
                    d_k, cg_info = cg(A_lam, b, maxiter=350,  callback = callback_counter, rtol = 1e-10)
                
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
