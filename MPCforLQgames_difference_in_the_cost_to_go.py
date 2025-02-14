import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la

"""
Computing the Nash equilibrium for a finite horizon (FH) linear quadratic (LQ) game. Then we take the time horizon T to infinite to
see what is the equivalent Nash equilibrium in the infinite horizon (IH) setting
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

# compute the algebraic Riccati equation with cross terms
def ARE_crossterms(A,Bi,Bj,Ki,Kj,Q,R,Rj,P):

    newP = Q + (Ki.T) @ R @ Ki + (Kj.T) @ Rj @ Kj + ( ( A - Bi @ Ki - Bj @ Kj ).T ) @ P @ ( A - Bi @ Ki - Bj @ Kj )

    return newP

# compute the Nash equilibrium of the finite-horizon game
def NE(A,B1,B2,Q1,Q2,R1,R2,P1,P2,T):

    K = np.zeros((T,2))

    for t in range(0,T,1):

        invM = matrixInverted(B1,B2,R1,R2,P1,P2)

        # Nash equilibrium policies at time t
        Kt = invM @ np.concatenate( ( (B1.T) @ P1 @ A, (B2.T) @ P2 @ A ) , axis = 0)
        K1t = np.array( [Kt[0,:]] )
        K2t = np.array( [Kt[1,:]] )

        # P matrix for the time t-1
        # P1 = ARE(A,B1,B2,K1t,K2t,Q1,R1,P1)
        # P2 = ARE(A,B2,B1,K2t,K1t,Q2,R2,P2)
        Rj = np.array([[0.3]])
        P1 = ARE_crossterms(A,B1,B2,K1t,K2t,Q1,R1,Rj,P1)
        P2 = ARE_crossterms(A,B2,B1,K2t,K1t,Q2,R2,Rj,P2)

        # update the Nash equilibrium policies
        K[T - t - 1,0] = Kt[0,0]
        K[T - t - 1,1] = Kt[1,0] 

    return K

# compute the cost in a receding horizon game where only the first policy is iteratively applied
def compute_RHG_costs(A,B1,B2,Q1,Q2,R1,R2,K1,K2):

    P1 = la.solve_discrete_are(A - B2 @ K2, B1, Q1, R1)
    P2 = la.solve_discrete_are(A - B1 @ K1, B2, Q2, R2)

    return [P1, P2]

#__________________________________________________________
#### Parameters

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
R2 = np.array([[1]])

# Time horizon
T = 50


#__________________________________________________________
#### Compute the Nash equilibria of the infinite-horizon game

P1correct = np.array([[7.2238087]])
P2correct = np.array([[7.2238087]])

# P1correct = np.array([[8.81072958]])
# P2correct = np.array([[6.82266124]])

#__________________________________________________________
#### First case

P1 = P1correct
P2 = P2correct

# Nash equilibrium policies
Kfirst = NE(A,B1,B2,Q1,Q2,R1,R2,P1,P2,T)

# Costs for a receding horizon games
K1 = np.array([[Kfirst[0,0]]])
K2 = np.array([[Kfirst[0,1]]])
[P1, P2] = compute_RHG_costs(A,B1,B2,Q1,Q2,R1,R2,K1,K2)
print("The first two policies are:\n", K1, K2)
print("The costs in the first case are:\n", P1, P2)
print("-------------------------------------------------------------------------------------")

#__________________________________________________________
#### Second case

P1 = P1correct + 1
P2 = P2correct

# Nash equilibrium policies
Ksecond = NE(A,B1,B2,Q1,Q2,R1,R2,P1,P2,T)

# Costs for a receding horizon games
K1 = np.array([[Ksecond[0,0]]])
K2 = np.array([[Ksecond[0,1]]])
[P1, P2] = compute_RHG_costs(A,B1,B2,Q1,Q2,R1,R2,K1,K2)
print("The first two policies are:\n", K1, K2)
print("The costs in the second case are:\n", P1, P2)
print("-------------------------------------------------------------------------------------")

#__________________________________________________________
#### Third case case

P1 = P1correct
P2 = P2correct + 1

# Nash equilibrium policies
Kthird = NE(A,B1,B2,Q1,Q2,R1,R2,P1,P2,T)

# Costs for a receding horizon games
K1 = np.array([[Kthird[0,0]]])
K2 = np.array([[Kthird[0,1]]])
[P1, P2] = compute_RHG_costs(A,B1,B2,Q1,Q2,R1,R2,K1,K2)
print("The first two policies are:\n", K1, K2)
print("The costs in the third case are:\n", P1, P2)
print("-------------------------------------------------------------------------------------")

#__________________________________________________________
#### Plot


x = np.linspace(0, T, T)

# Define colors for each case
colors = ['b', 'g', 'r']  # First, Second, and Third cases

# Create figure and axes
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

# Player 1 plot
axs[0].plot(x, Kfirst[:, 0], '.', label='k1 - First case', color=colors[0])
axs[0].plot(x, Ksecond[:, 0], '.', label='k1 - Second case', color=colors[1])
axs[0].plot(x, Kthird[:, 0], '.', label='k1 - Third case', color=colors[2])
axs[0].set_title('Player 1')
axs[0].set_xlabel('Time step t')
axs[0].set_ylabel('k1')
axs[0].grid(True)
axs[0].set_ylim(0, A)
axs[0].legend()

# Player 2 plot
axs[1].plot(x, Kfirst[:, 1], '.', label='k2 - First case', color=colors[0])
axs[1].plot(x, Ksecond[:, 1], '.', label='k2 - Second case', color=colors[1])
axs[1].plot(x, Kthird[:, 1], '.', label='k2 - Third case', color=colors[2])
axs[1].set_title('Player 2')
axs[1].set_xlabel('Time step t')
axs[1].set_ylabel('k2')
axs[1].grid(True)
axs[1].set_ylim(0, A)
axs[1].legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()




# # Figure - there are two subplots, one for each agent, put side by side
# fig, axs = plt.subplots(3, 2, figsize=(10, 8))
# x = np.linspace(0, T, T)

# # First agent - First case
# axs[0,0].plot(x, Kfirst[:,0], '.', label='k1', color='b')
# axs[0,0].set_title('Plot of k1 - First case')
# axs[0,0].set_xlabel('Time step t')
# axs[0,0].set_ylabel('k1')

# # Second agent - First case
# axs[0,1].plot(x, Kfirst[:,1], '.', label='k2', color='r')
# axs[0,1].set_title('Plot of k2 - First case')
# axs[0,1].set_xlabel('Time step t')
# axs[0,1].set_ylabel('k2')

# # First agent - Second case
# axs[1,0].plot(x, Ksecond[:,0], '.', label='k1', color='b')
# axs[1,0].set_title('Plot of k1 - Second case')
# axs[1,0].set_xlabel('Time step t')
# axs[1,0].set_ylabel('k1')

# # Second agent - Second case
# axs[1,1].plot(x, Ksecond[:,1], '.', label='k2', color='r')
# axs[1,1].set_title('Plot of k2 - Second case')
# axs[1,1].set_xlabel('Time step t')
# axs[1,1].set_ylabel('k2')

# # First agent - Third case
# axs[2,0].plot(x, Kthird[:,0], '.', label='k1', color='b')
# axs[2,0].set_title('Plot of k1 - Third case')
# axs[2,0].set_xlabel('Time step t')
# axs[2,0].set_ylabel('k1')

# # Second agent - Third case
# axs[2,1].plot(x, Kthird[:,1], '.', label='k2', color='r')
# axs[2,1].set_title('Plot of k2 - Third case')
# axs[2,1].set_xlabel('Time step t')
# axs[2,1].set_ylabel('k2')

# # Settings
# for i in range(3):
#     axs[i, 0].grid(True)
#     axs[i, 0].set_ylim(0,A)
#     axs[i, 1].grid(True)
#     axs[i, 1].set_ylim(0,A)

# # Plot 
# plt.tight_layout()  # Adjust the space between subplots
# plt.show()