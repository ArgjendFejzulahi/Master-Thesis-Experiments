from scipy.linalg import qr, cholesky, svd, solve_triangular
from scipy.sparse import eye
from scipy.sparse.linalg import aslinearoperator, LinearOperator, eigsh
import numpy as np


""" This module containts randomized algorithms specifically the randomized nyström approximation and preconditioner.

"""

def randomPowerErrEst(A, U, Lambda, q): 
    n, d = np.shape(A)
    g = np.random.randn(n,1)
    g_norm = np.linalg.norm(g)
    v_0 = g/g_norm
    for i in range(0,q):
        v = A @ v_0 - U @ (np.diag(Lambda) @ ( U.T @ v_0)) 
        E_hat = np.dot(v_0.T, v)
        v = v / np.linalg.norm(v)
        v_0 = v
        
    return np.abs(E_hat[0])

def cond_2_sparse(A: any): 
    """Fast method to calculate the condition number of a symmetric Matrix A (or complex hermitian)"""
    
    largest_eigenvalue = eigsh(A, k=1, which='LM', return_eigenvectors=False)[0]
    smallest_eigenvalue = eigsh(A, k=1, which='SM', return_eigenvectors=False)[0]
    
    return largest_eigenvalue / smallest_eigenvalue

class Nyström:
    
    def __init__(self, A: LinearOperator, mu: float): 
        self.A = A
        self.mu = mu
        self.n = self.A.shape[0]
        self.U = None
        self.Lambda = None 
        
    def approximation(self, r): 
        """Generate and store the Nyström approximation components U and Lambda.
        
        r: low-rank to approximate: """
        
        Omega = np.random.randn(self.n, r)
        Omega, _ = qr(Omega, mode='economic')
        
        Y = np.empty((self.n, r))
        for i in range(r):
            Y[:, i] = self.A.matvec(Omega[:, i])

        nu = np.finfo(float).eps * np.linalg.norm(Y, 'fro')
        Y_nu = Y + nu * Omega

        B = Omega.T @ Y_nu
        C = cholesky(B, lower=True)
        B = solve_triangular(C, Y_nu.T, trans=0, lower=True, check_finite=False).T

        U, Sigma, _ = svd(B, full_matrices=False)
        Lambda = np.maximum(0, Sigma**2 - nu * np.eye(r))

        self.U = U
        self.Lambda = Lambda

    def adaptive_approximation(self, l0: int, lmax: int, tau: float):
        
        """Adaptive approximation method to compute and store U and Lambda.
        l0: initital start rank
        lmax: maximum rank
        tau: specific parameter (see Masterthesis)"""
        
        tol_err = tau * self.mu
        tol_rat = (tau * self.mu) / 10
        Y = np.empty((self.n, 0))
        Omega = np.empty((self.n, 0))
        Err = 1e8
        lambda_l = 1e8
        m = l0

        while (Err > tol_err) & (lambda_l / self.mu > tol_rat):
            Omega_0 = np.random.randn(self.n, m)
            Omega_0, _ = qr(Omega_0, mode='economic')
            
            Y_0 = np.empty((self.n, m))
            for i in range(m):
                Y_0[:, i] = self.A.matvec(Omega_0[:, i])
            
            Omega = np.concatenate((Omega, Omega_0), axis=1)
            Y = np.concatenate((Y, Y_0), axis=1)

            nu = np.sqrt(self.n) * np.finfo(float).eps * np.linalg.norm(Y, 2)
            Y_nu = Y + nu * Omega

            B = Omega.T @ Y_nu
            C = cholesky(B, lower=True)
            B = solve_triangular(C, Y_nu.T, trans=0, lower=True, check_finite=False).T

            U, Sigma, _ = svd(B, full_matrices=False)
            Lambda = np.maximum(0, Sigma**2 - nu)

            lambda_l = np.min(Lambda)
            Err = randomPowerErrEst(self.A, U, Lambda, 150)
            m = l0
            l0 *= 2

            if l0 > lmax:
                l0 = l0 - m
                m = lmax - l0
                Omega_0 = np.random.randn(self.n, m)
                Omega_0, _ = qr(Omega_0, mode='economic')
                Y_0 = np.empty((self.n, m))
                for i in range(m):
                    Y_0[:, i] = self.A.matvec(Omega_0[:, i])

                Omega = np.concatenate((Omega, Omega_0), axis=1)
                Y = np.concatenate((Y, Y_0), axis=1)

                nu = np.sqrt(self.n) * np.finfo(float).eps * np.linalg.norm(Y, 2)
                Y_nu = Y + nu * Omega

                B = Omega.T @ Y_nu
                C = cholesky(B, lower=True)
                B = solve_triangular(C, Y_nu.T, trans=0, lower=True, check_finite=False).T

                U, Sigma, _ = svd(B, full_matrices=False)
                Lambda = np.maximum(0, Sigma**2 - nu)

                self.U = U
                self.Lambda = Lambda
                
                return

        self.U = U
        self.Lambda = Lambda

    
    ### either transform directly to linearoperator or create a custum one
    def preconditioner(self):
        """Return the Nyström-based preconditioner using stored U and Lambda."""
        if self.U is None or self.Lambda is None:
            raise ValueError("U and Lambda have not been computed. Run 'approximation(r)' or 'adaptive_approximation(l0, lmax, tau)' first.")
        
        # Ensure Lambda is a vector, not a matrix ????
        lambda_diag = np.diag(self.Lambda) if self.Lambda.ndim > 1 else self.Lambda
        lambda_l = np.min(lambda_diag)

        Lambda_inv_mu = np.diag(1 / (lambda_diag + self.mu))

        U_op = aslinearoperator(self.U)
        Lambda_inv_mu_op = aslinearoperator(Lambda_inv_mu)

        UUT = U_op @ U_op.T

        P_inv = (lambda_l + self.mu) * U_op @ Lambda_inv_mu_op @ U_op.T + (aslinearoperator(eye(self.n)) - UUT)

        return P_inv

if __name__ == "__main__":
    
    from scipy.sparse.linalg import LinearOperator
    import numpy as np

    # Define a test symmetric positive definite matrix (A = X @ X.T)
    np.random.seed(42)
    n = 100  # dimension
    X = np.random.randn(n, n)
    A_mat = X @ X.T

    # Define LinearOperator from the matrix
    def matvec(v):
        return A_mat @ v

    A = LinearOperator(shape=(n, n), matvec=matvec, dtype=A_mat.dtype)

    # Regularization parameter
    mu = 1e-5

    # Create Nyström object
    nys = Nyström(A, mu)

    # Test standard approximation
    r = 20
    print(f"\nTesting approximation with rank {r}...")
    nys.approximation(r)
    print("U shape:", nys.U.shape)
    print("Lambda:", nys.Lambda)

    # Test adaptive approximation
    l0 = 10
    lmax = 50
    tau = 35
    print(f"\nTesting adaptive approximation with l0={l0}, lmax={lmax}, tau={tau}...")
    nys.adaptive_approximation(l0, lmax, tau)
    print("Adaptive U shape:", nys.U.shape)
    print("Adaptive Lambda:", nys.Lambda)

    # Test preconditioner
    try:
        print("\nTesting preconditioner...")
        P = nys.preconditioner()
        # Apply the preconditioner to a random vector
        x = np.random.randn(n)
        Px = P @ x
        print("Preconditioner application successful. Shape of Px:", Px.shape)
    except Exception as e:
        print("Error during preconditioner test:", e)
