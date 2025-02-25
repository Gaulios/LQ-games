import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

"""
Applying gradient descent in potential decoupled linear quadratic games.

System evolution
x_{t+1} = A * x_{t} + B^1 u^1_t + u^2_t

Linear state feedback control
u_{i,t} = k_i x_{t}

Cost function
J_i ( k_1, k_2 ) = sum_{ t = 0 }^\infty ( q_i * ( x_t ** 2 ) + r_i * ( u_{i,t} ** 2 ) )

"""

##########################################################################################################

def perform_gradient_descent( data, step_size, num_iterations):
    
    N = data.N
    n = data.n
    d = data.d
    T = data.T
    A = data.A
    B = data.B
    Q = data.Q
    R = data.R

    # Initialize P matrices
    P = np.zeros((N*n, N*n, T + 1))
    P[:, :, -1] = Q

    # Initialize K as a 3D array (block diagonal matrices for each time step)
    K = np.zeros((N * d, N * n, T))
    # Generate K_t for each time step t and store in K[:, :, t]
    for t in range(T):
        K_blocks = [np.random.randn(d, n) for _ in range(N)]  # Generate random blocks
        K[:, :, t] = block_diag(*K_blocks)  # Create block diagonal matrix

    mask_blocks = [np.ones((d, n)) for _ in range(N)]
    mask = block_diag(*mask_blocks)

    K_history = np.zeros((num_iterations, N * d, N * n, T))
    norm_grad_descent = np.zeros(num_iterations)
    E = np.zeros(( N * d, N * n))

    # Gradient descent loop
    for m in range(num_iterations):

        for t in range(T - 1, -1, -1):
            A_t = A - B @ K[ :, :, t]
            
            # Compute P matrices
            P[:, :, t] = ( Q[:, :] + K[:, :, t].T @ R[:, :] @ K[:, :, t] + A_t.T @ P[:, :, t + 1] @ A_t )

            # Compute E matrices - the gradient is equal to 2*E*Sigma_K
            E[:, :] = R[:, :] @ K[:, :, t] - B.T @ P[:, :, t + 1] @ A_t
            
            # Update policies
            K[:, :, t] -= 2 * step_size * ( mask * E )            

        # Store history
        K_history[m, :, :, :] = K
        norm_grad_descent[m] = np.linalg.norm(mask * E)

    # print("The gradient is:\n", E)
    # print("\n The last K is:\n", K)
    # print("--------------------------------------------------")
    return K, K_history, norm_grad_descent


def print_gradient_descent_norm(all_gradient_norms):

    num_runs = all_gradient_norms.shape[0]

    # Plot results for each run and each player
    plt.figure(figsize=(10, 6))
    for r in range(num_runs):
        plt.plot(all_gradient_norms[r, :], label=f'Run {r+1}')

    ax = plt.gca()
    ax.set_xlim([1, num_iterations - 100])
    # ax.set_ylim([10**(-8), 10**4])

    plt.xlabel('Iteration')
    plt.ylabel('Norm Difference')
    plt.title('Gradient Descent Convergence Across Runs')
    #plt.legend(loc='upper right', fontsize=8, ncol=2)  # Adjust legend size
    plt.xscale('log')  # Log scale for better visibility
    plt.yscale('log')  # Log scale for better visibility
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


    return None

def print_difference_norm(num_iterations,all_K_history, K):
    
    num_runs = all_K_history.shape[0]  # Number of gradient descent runs
    norm_diffs = np.zeros((num_runs, num_iterations - 100))  # Store norms for each run and iteration

    for r in range(num_runs):  # Iterate over different K_history runs
        K_history = all_K_history[r]  # Extract single run of K_history
        for m in range(num_iterations - 100):
            norm_diffs[r, m] = np.linalg.norm(K_history[m, :, :, :] - K)  # Compute norm diff

    # Plot results for each run and each player
    plt.figure(figsize=(10, 6))
    for r in range(num_runs):
        plt.plot(norm_diffs[r, :], label=f'Run {r+1}')

    plt.xlabel('Iteration')
    plt.ylabel('Norm Difference')
    plt.title('Gradient Descent Convergence Across Runs')
    plt.legend(loc='upper right', fontsize=8, ncol=2)  # Adjust legend size
    plt.xscale('log')  # Log scale for better visibility
    plt.yscale('log')  # Log scale for better visibility
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()

    return None


def print_convergence_to_single_point( num_runs, final_K_values, ):
    # Compute norm differences between final K values across runs
    norm_diffs = np.zeros((num_runs, num_runs))

    for i in range(num_runs):
        for j in range(num_runs):
            norm_diffs[i, j] = np.linalg.norm(final_K_values[i, :, :, :] - final_K_values[j, :, :, :])

    # Print mean difference (should be small if all runs converge to same K)
    print("\nMean norm difference across final K values:", np.mean(norm_diffs))

    # Plot norm differences
    plt.imshow(norm_diffs, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Norm Difference")
    plt.title("Norm Differences Between Final Policies in Different Runs")
    plt.xlabel("Run Index")
    plt.ylabel("Run Index")
    plt.show()

    return None


#################################################################################################

class GameData:
    def __init__(self, N, n, d, T, A, B, Q, R):
        self.N = N
        self.n = n
        self.d = d
        self.T = T
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R


#################################################################################################


# Problem data definition
n = 1  # state dimension
d = 1  # input dimension
N = 3   # number of players
T = 3  # time horizon

# Random stable dynamics matrix
# Generate N random n x n matrices
A_blocks = [np.random.randn(n, n) for _ in range(N)]
# Create the block diagonal matrix
A = block_diag(*A_blocks)
#A = 0.95 * A / max(abs(np.linalg.eigvals(A)))

# Random input matrix
# Generate N random n x n matrices
B_blocks = [np.random.randn(n, d) for _ in range(N)]
# Create the block diagonal matrix
B = block_diag(*B_blocks)
#B = np.random.randn(n, d, N)

# Cost matrices - Potential cost 
Q = np.random.randn(N*n, N*n)
Q = Q.T @ Q  # Ensure positive semi-definiteness
R = np.random.randn(N*d, N*d)
R = R.T @ R + 0.01*np.eye(N*d)  # Ensure positive definiteness


data = GameData(N, n, d, T, A, B, Q, R)

print("--------------------------------------------------")
print("The value of A is:\n", A)
print("--------------------------------------------------")
print("The value of B is:\n", B)
print("--------------------------------------------------")
print("The value of Q is:\n", Q)
print("--------------------------------------------------")
print("The value of R is:\n", R)
print("--------------------------------------------------")


#################################################################################################

# Policy Gradient parameters
num_runs = 50  # Number of random initializations
num_iterations = int(1e3)
step_size = 1e-2
final_K = np.zeros((num_runs, N*d, N*n, T))
all_K_history = np.zeros((num_runs, num_iterations, N * d, N * n, T))
all_gradient_norms = np.zeros((num_runs, num_iterations))

# Repeat gradient descent from different initializations
for run in range(num_runs):
    
    print(f"Run {run+1}/{num_runs}")
    final_K[run, :, :, :], all_K_history[run, :, :, :, :], all_gradient_norms[run, :] = perform_gradient_descent( data, step_size, num_iterations)


####################################################################################################

print_convergence_to_single_point( num_runs, final_K, )
average_final_K = (1 / run)*np.sum(final_K, axis=0)
print_gradient_descent_norm(all_gradient_norms)
