import numpy as np
import matplotlib.pyplot as plt
import json

class GameData:
    def __init__(self, A, B, Q, R, P):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.P = P

    def to_dict(self):
        return {
            'A': self.A.tolist(),
            'B': self.B.tolist(),
            'Q': self.Q.tolist(),
            'R': self.R.tolist(),
            'P': self.P.tolist()
        }

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f)

def generateRandomGame(n,d,N):
    # Random dynamics matrix
    A = np.random.randn(n,n)

    # Random input matrix
    B = np.random.randn(N, n, d)

    # Cost matrices
    Q = np.zeros((N, n, n))
    R = np.zeros((N, d, d))
    P = np.zeros((N, n, n)) # terminal cost
    for i in range(N):
        Qtemp = np.random.randn(n, n)
        Q[i] = Qtemp.T @ Qtemp # Ensure positive semi-definiteness
        Rtemp = np.random.randn(d, d)
        R[i] = Rtemp.T @ Rtemp + 0.01*np.eye(d)  # Ensure positive definiteness
        Ptemp = np.random.randn(n, n)
        P[i] = Ptemp.T @ Ptemp # Ensure positive semi-definiteness
    
    game = GameData(A, B, Q, R, P)
    return game

def check_symmetric_submatrices(M, tol=1e-8):
    """
    Check that all submatrices M[i] are symmetric within a given tolerance.

    Parameters:
        M (np.ndarray): A 3D numpy array of shape (N, a, a).
        tol (float): The tolerance level for comparing symmetry. Defaults to 1e-8.

    Returns:
        bool: True if all submatrices are symmetric within the tolerance, False otherwise.
    """
    N, a, b = M.shape
    if a != b:
        raise ValueError("Each submatrix must be square (second and third dimensions must be equal).")
    
    for i in range(N):
        if not np.allclose(M[i], M[i].T, atol=tol):
            print(f"Submatrix {i} is not symmetric.")
            return False
    return True

def concatenateBPA(N, A, B, P_time):
    """
    Costruisce la matrice concatenata che contiene i blocchi B[i].T @ P_time[i] @ A per ogni agente.
    A: (n, n), B: (N, n, d), P_time: (N, n, n)
    Output: matrice di forma (N*d, n)
    """
    n = A.shape[0]
    conc = np.zeros((N*d, n))
    for i in range(N):
        conc[i*d:(i+1)*d, :] = B[i].T @ P_time[i] @ A
    return conc

def convergenceK(T, K):
    conv = 1e-10*np.ones((1, T-1))
    for t in range(1, T):
        conv[0, t-1] = np.linalg.norm(K[:, T - t - 1, :, :] - K[:, T - t, :, :])
        if conv[0, t-1] < 1e-9:
            break
    z = 50
    exists = conv[0, T//2:].sum() > z
    return conv, exists

def matrixInverted(N, d, B, R, P_time):
    """
    Calcola l'inversa della matrice M costruita a blocchi.
    B: (N, n, d), R: (N, d, d), P_time: (N, n, n)
    """
    M = np.zeros((N*d, N*d))
    for i in range(N):
        for j in range(N):
            if i == j:
                M[i*d:(i+1)*d, i*d:(i+1)*d] = B[i].T @ P_time[i] @ B[i] + R[i]
            else:
                M[i*d:(i+1)*d, j*d:(j+1)*d] = B[i].T @ P_time[i] @ B[j]
    if np.abs(np.linalg.det(M)) < 0.001:
        print("ATTENTION: determinant very small", np.linalg.det(M))
    return np.linalg.inv(M)

def ARE(n, N, A, B, Q, R, Ksplit, P_time):
    """
    Calcola la soluzione dell'equazione di Riccati per ciascun agente.
    A: (n, n)
    B: (N, n, d)
    Q: (N, n, n)
    R: (N, d, d)
    Ksplit: (N, d, n)
    P_time: (N, n, n)
    """
    A_closed = A.copy()
    for i in range(N):
        A_closed = A_closed - B[i] @ Ksplit[i]
    newP = np.zeros((N, n, n))
    for i in range(N):
        newP[i] = Q[i] + Ksplit[i].T @ R[i] @ Ksplit[i] + A_closed.T @ P_time[i] @ A_closed
    return newP

def ARE_crossterms(A, Bi, Bj, Ki, Kj, Q, R, Rj, P):
    newP = Q + Ki.T @ R @ Ki + Kj.T @ Rj @ Kj + (A - Bi @ Ki - Bj @ Kj).T @ P @ (A - Bi @ Ki - Bj @ Kj)
    return newP

def NE(n, d, N, T, game, show):
    A = game.A            # A: (n, n)
    B = game.B            # B: now shape (N, n, d)
    Q = game.Q            # Q: (N, n, n)
    R = game.R            # R: (N, d, d)
    P = game.P            # P: (N, n, n)
    
    # K avrà forma (N, T, d, n) e Pt (N, T+1, n, n)
    K = np.zeros((N, T, d, n))
    Pt = np.zeros((N, T+1, n, n))
    Pt[:, T, :, :] = P  # inizialmente Pt[:, T] = P
    
    for t in range(0, T):
        current_time = T - t
        invM = matrixInverted(N, d, B, R, Pt[:, current_time, :, :])
        conc = concatenateBPA(N, A, B, Pt[:, current_time, :, :])
        if show == 1:
            print("--------------------")
            print("At time", current_time, "the matrix invM is\n", invM)
            print("At time", current_time, "the matrix conc is\n", conc)
            print("--------------------")
        # Kt ha forma (N*d, n)
        Kt = invM @ conc
        Ksplit = np.zeros((N, d, n))
        for i in range(N):
            Ksplit[i] = Kt[i*d:(i+1)*d, :]
        # Aggiorna la matrice Pt per il tempo precedente
        Pt[:, current_time - 1, :, :] = ARE(n, N, A, B, Q, R, Ksplit, Pt[:, current_time, :, :])
        K[:, current_time - 1, :, :] = Ksplit
    return K

def plot_convergence(num_runs, T, all_conv):
    plt.figure(figsize=(10, 5))
    for i in range(num_runs):
        plt.plot(range(T-1), all_conv[i, :], alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    ax.set_xlim([1, T-2])
    ax.set_ylim([10**(-8), 10**4])
    plt.grid()
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Evolution of Convergence Across Runs')
    plt.show()

def plot_evolution_K(K):
    """
    Plotta l'evoluzione di ogni elemento della matrice di politiche di equilibrio di Nash K.
    K: now shape (N, T, d, n)
    """
    N, T, d, n = K.shape
    total_cols = d * n
    fig, axes = plt.subplots(nrows=N, ncols=total_cols, figsize=(total_cols*3, N*2))
    if N == 1:
        axes = np.atleast_2d(axes)
    if total_cols == 1:
        axes = axes[:, np.newaxis]
    for i in range(N):
        for r in range(d):
            for c in range(n):
                col = r * n + c
                ax = axes[i, col]
                ax.plot(np.arange(T), K[i, :, r, c])
                ax.set_title(f'Agent {i+1}\nK[{r},{c}]', fontsize=8)
                ax.tick_params(labelsize=8)
                if i == N-1:
                    ax.set_xlabel('Time', fontsize=8)
    plt.tight_layout()
    plt.show()

def find_pattern_length(K, T):
    """
    Trova la lunghezza L del pattern ripetuto in K.
    K: now shape (N, T, d, n), viene verificato lungo l'asse del tempo (asse 1).
    """
    for L in range(1, T//2):
        if np.allclose(K[:, :L, :, :], K[:, L:2*L, :, :], atol=1e-8):
            return L
    return -1

def calculate_A_minus_BK(A, B, K, T):
    """
    Calcola A - B1 K1 - B2 K2 - ... per ogni istante di tempo.
    A: (n, n)
    B: (N, n, d)
    K: (N, T, d, n)
    """
    n = A.shape[0]
    A_minus_BK = np.zeros((n, n, T))
    for t in range(T):
        A_closed = A.copy()
        for i in range(N):
            A_closed -= B[i] @ K[i, t, :, :]
        A_minus_BK[:, :, t] = A_closed
    return A_minus_BK

def plot_A_minus_BK(A_minus_BK, L):
    n, _, T = A_minus_BK.shape
    fig, axes = plt.subplots(nrows=n, ncols=n, figsize=(n*3, n*3))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.plot(np.arange(2*L), A_minus_BK[i, j, :2*L])
            ax.set_title(f'( A - B K ) [{i},{j}]', fontsize=8)
            ax.tick_params(labelsize=8)
            if i == n-1:
                ax.set_xlabel('Time', fontsize=8)
            if j == 0:
                ax.set_ylabel('Value', fontsize=8)
    plt.tight_layout()
    plt.show()


############################################################################################################
# Problem data definition
n = 2  # state dimension
d = 1  # input dimension
N = 3  # number of players
T = 1000  # Time horizon
num_runs = 50
all_conv = np.zeros((num_runs, T-1))
num_conv = 0

############################################################################################################
for i in range(0, num_runs):

    # Generate random game
    game = generateRandomGame( n, d, N)

    # compute value iteration
    K = NE(n, d, N, T, game, 0)

    # compute convergence
    (all_conv[i, :], exists) = convergenceK(T, K)

    if exists == True:
        num_conv += 1


############################################################################################################
print(f"The percentage of non converging runs is {num_conv/num_runs*100}%")
plot_convergence(num_runs, T, all_conv)