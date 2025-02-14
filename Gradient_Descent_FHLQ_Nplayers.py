import numpy as np
import matplotlib.pyplot as plt

"""
<<<<<<< HEAD
Applying gradient descent in linear quadratic games.
=======
Applying gradient descent in potential decoupled linear quadratic games.
>>>>>>> 96f0a03aa1233cd57c0c56eb717ceab0291f37e2

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
    P = np.zeros((n, n, N, T + 1))
    for i in range(N):
        P[:, :, i, -1] = Q[:, :, i]

    # Initialize K matrices
    K = np.random.randn(d, n, N, T)*0.1
    K_history = np.zeros((num_iterations, d, n, N, T))
    E = np.zeros((d, n, N))

    # Gradient descent loop
    for m in range(num_iterations):

        for t in range(T - 1, -1, -1):
            A_bar = A.copy()
            for i in range(N):
                A_bar -= np.outer(B[:, :, i], K[:, :, i, t])
            
            # Compute P matrices
            for i in range(N):
                P[:, :, i, t] = (
                    Q[:, :, i] + K[:, :, i, t].T @ R[:, :, i] @ K[:, :, i, t] + A_bar.T @ P[:, :, i, t + 1] @ A_bar
                )

            # Compute E matrices - the gradient is equal to 2*E*Sigma_K
            for i in range(N):
                E[:, :, i] = R[:, :, i] @ K[:, :, i, t] - B[:, :, i].T @ P[:, :, i, t + 1] @ A_bar
            
            # Update policies
            for i in range(N):
                K[:, :, i, t] -= 2 * step_size * E[:, :, i]
            
        # Store history
        K_history[m, :, :, :, :] = K

    return K


def print_gradient_descent_norm( num_iterations, N, K_history, K ):

    norm_diffs = np.zeros((num_iterations - 100, N))  # Store norms for each iteration and player

    for m in range(num_iterations - 100):
        for i in range(N):
            norm_diffs[m, i] = np.linalg.norm(K_history[m, :, :, i, :] - K[:, :, i, :])

    # Print final norm differences
    for i in range(N):
        print(f"Player {i+1} final norm difference: {norm_diffs[-2, i]}")

    for i in range(N):
        plt.plot(norm_diffs[:, i], label=f'Player {i+1}')

    plt.xlabel('Iteration')
    plt.ylabel('Norm Difference')
    plt.title('Gradient Descent Convergence')
    plt.legend()
    plt.yscale('log')  # Optional: Use log scale for better visibility
    plt.show()

    return None


def print_convergence_to_single_point( num_runs, final_K_values, ):
    # Compute norm differences between final K values across runs
    norm_diffs = np.zeros((num_runs, num_runs))

    for i in range(num_runs):
        for j in range(num_runs):
            norm_diffs[i, j] = np.linalg.norm(final_K_values[i, :, :, :, :] - final_K_values[j, :, :, :, :])

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
n = 2  # state dimension
d = 1  # input dimension
N = 4   # number of players
T = 5  # time horizon

# Random stable dynamics matrix
A = np.random.randn(n, n)
A = 0.95 * A / max(abs(np.linalg.eigvals(A)))

# Random input matrix
B = np.random.randn(n, d, N)

# Cost matrices
Q = np.zeros((n, n, N))
R = np.zeros((d, d, N))  
for i in range(N):
    Q[:, :, i] = np.random.randn(n, n)
    Q[:, :, i] = Q[:, :, i].T @ Q[:, :, i]  # Ensure positive semi-definiteness
    R[:, :, i] = np.random.randn(d, d)
    R[:, :, i] = R[:, :, i].T @ R[:, :, i]  # Ensure positive semi-definiteness


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
num_runs = 10  # Number of random initializations
num_iterations = int(1e4)
step_size = 1e-3
final_K_values = np.zeros((num_iterations, d, n, N, T))


# Repeat gradient descent from different initializations
for run in range(num_runs):
    print(f"Run {run+1}/{num_runs}")

    #K_history = perform_gradient_descent( data, step_size, num_iterations)

    # Store final K for this run
    #final_K_values[run, :, :, :, :] = K_history[num_iterations - 1, :, :, :, :]
    final_K_values[run, :, :, :, :] = perform_gradient_descent( data, step_size, num_iterations)


####################################################################################################

print_convergence_to_single_point( num_runs, final_K_values, )