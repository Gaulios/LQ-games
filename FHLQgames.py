import numpy as np
import matplotlib.pyplot as plt

"""
Computing the Nash equilibrium for a finite horizon (FH) discrete time linear quadratic (LQ) game with two agents.
We compute it by applying dynamic programming. At each time step, starting from the last one, 
we need to solve a linear system given by the following two equations:

1. (B1.T @ P1 @ B1 + R1) @ K1 + (B1.T @ P1 @ B2) @ K2 = (B1.T) @ P1 @ A

2. (B2.T @ P2 @ B1) @ K1 + (B2.T @ P2 @ B2 + R2) @ K2 = (B2.T) @ P2 @ A

Giulio Salizzoni @ EPFL
2024 June
"""

# compute the inverse of the matrix M. It is needed to solve the linear system that gives us the Nash equilibrium policies
def matrixInverted(B1,B2,R1,R2,P1,P2):

    M = np.concatenate((
    np.concatenate((B1.T @ P1 @ B1 + R1, B1.T @ P1 @ B2), axis=1),
    np.concatenate((B2.T @ P2 @ B1, B2.T @ P2 @ B2 + R2), axis=1)
    ), axis=0)

    return np.linalg.inv(M)

# compute the algebraic Riccati equation
def ARE(A,Bi,Bj,Ki,Kj,Q,R,P):

    newP = Q + (Ki.T) @ R @ Ki + ( ( A - Bi @ Ki - Bj @ Kj ).T ) @ P @ ( A - Bi @ Ki - Bj @ Kj )

    return newP
    

# number of agents
#n = 2

# system matrices
A = np.array([[5]])
B1 = np.array([[1]])
B2 = np.array([[1]])

# parameters of the cost function of the first agent
Q1 = np.array([[1]])
R1 = np.array([[1]])

# parameters of the cost function of the second agent
Q2 = np.array([[1]])
R2 = np.array([[2]])

# # Definisci le matrici A, B1 e B2
# A = np.array([[0.588, 0.028],
#               [0.570, 0.056]])

# B1 = np.array([[1], 
#                [1]])

# B2 = np.array([[0], 
#                [1]])

# # Definisci Q1 e Q2 utilizzando diag
# Q1 = np.diag([1, 1])
# Q2 = np.diag([1, 2])

# # Definisci R1 e R2
# R1 = np.array([[0.01]])
# R2 = R1

# Time horizon
T = 50

# Nash equilibrium policies
K = np.zeros((T,2))

# Define the final matrices P1 and P2
#P1 = Q1
#P2 = Q2
P1 = np.array([[6.614]])
P2 = np.array([[14.722]])

for t in range(T-1,-1,-1):

    invM = matrixInverted(B1,B2,R1,R2,P1,P2)
    #print("The inverse of M is:\n", invM)

    # Nash equilibrium policies at time t
    Kt = invM @ np.concatenate( ( (B1.T) @ P1 @ A, (B2.T) @ P2 @ A ) , axis = 0)
    #print("The NE strategies at time", t, "are:\n", invM)
    K1t = np.array( [Kt[0,:]] )
    K2t = np.array( [Kt[1,:]] )

    # P matrix for the time t-1
    P1 = ARE(A,B1,B2,K1t,K2t,Q1,R1,P1)
    P2 = ARE(A,B2,B1,K2t,K1t,Q2,R2,P2)

    # update the Nash equilibrium policies
    K[t,0] = Kt[0,0]
    K[t,1] = Kt[1,0]



# Figure - there are two subplots, one for each agent, put side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
x = np.linspace(0, T, T)

# First agent
axs[0].plot(x, K[:,0], '.', label='k1', color='b')
axs[0].set_title('Plot of k1')
axs[0].set_xlabel('Time step t')
axs[0].set_ylabel('k1')
axs[0].grid(True)
axs[0].set_ylim(0,5)

# Second agent
axs[1].plot(x, K[:,1], '.', label='k2', color='r')
axs[1].set_title('Plot of k2')
axs[1].set_xlabel('Time step t')
axs[1].set_ylabel('k2')
axs[1].grid(True)
axs[1].set_ylim(0,5)

# Plot 
plt.tight_layout()  # Adjust the space between subplots
plt.show()


# Print the value of the policies at t = 0 and t = 80
print("The NE at time t = 0 are:\n", K[0,:])
print("The NE at time t =80 are:\n", K[80,:])