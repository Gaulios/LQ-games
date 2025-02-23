import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import colormaps as cmaps

"""
Computing the Nash equilibrium for a finite horizon (FH) linear quadratic (LQ) game. Then we take the time horizon T to infinite to
see what is the equivalent Nash equilibrium in the infinite horizon (IH) setting
"""


def compute_polynomial_roots(a,q1,q2,r1,r2):

    v1 = a*(r1**2)*(r2**2)
    v2 = -2.5*(a**2)*(r1**2)*(r2**2) + q2*(r1**2)*r2 - q1*r1*(r2**2) + (r1**2)*(r2**2)
    v3 = 2*(a**3)*(r1**2)*(r2**2) - 2*a*q2*(r1**2)*r2 + 2*a*q1*r1*(r2**2) - 2*a*(r1**2)*(r2**2)
    v4 = -0.5*(a**4)*(r1**2)*(r2**2) + (a**2)*q2*(r1**2)*r2 - (a**2)*q1*r1*(r2**2) + (a**2)*(r1**2)*(r2**2) + 0.5*(q2**2)*(r1**2) - 0.5*(q1**2)*(r2**2) - q1*r1*(r2**2) - 0.5*(r1**2)*(r2**2)
    v5 = - a*(q2**2)*(r1**2)
    v6 = 0.5*(a**2)*(q2**2)*(r1**2)

    roots = np.roots([v1,v2,v3,v4,v5,v6])
    return roots

def compute_best_response( a, q, r, kj ):

    temp = r*( 1 - ( a - kj )**2 ) + q
    ki = ( 2*( a - kj )*q )/( np.sqrt( temp**2 + 4*r*( ( a - kj )**2 )*q )  + temp )

    return ki

def compute_P( a, q, r, ki, kj ):

    cost = ( q + r*( ki**2 ) )/( 1 - ( a - ki - kj )**2 )

    return cost

def compute_gradient( a, q, r, ki, kj ): 

    grad = ( 2*r*( a - kj )*( ki**2 ) + 2*( r + q - r*( ( a - kj )**2 ) )*ki - 2*( a - kj )*q )/( ( 1 - ( a - ki - kj )**2 )**2 )

    return grad

def compute_second_derivative( a, q, r, ki, kj):

    second_derivative = 4*ki*r*(-2*a + 2*ki + 2*kj)/(1 - (a - ki - kj)**2)**2 + 2*r/(1 - (a - ki - kj)**2) + 2*(ki**2*r + q)/(1 - (a - ki - kj)**2)**2 + (ki**2*r + q)*(-4*a + 4*ki + 4*kj)*(-2*a + 2*ki + 2*kj)/(1 - (a - ki - kj)**2)**3
    
    return second_derivative

def compute_mixed_derivative( a, q, r, ki, kj):

    mixed_derivative = 2*ki*r*(-2*a + 2*ki + 2*kj)/(1 - (a - ki - kj)**2)**2 + 2*(ki**2*r + q)/(1 - (a - ki - kj)**2)**2 + (ki**2*r + q)*(-4*a + 4*ki + 4*kj)*(-2*a + 2*ki + 2*kj)/(1 - (a - ki - kj)**2)**3
    
    return mixed_derivative

def show_Nash_equilibrium( a, k, cost, grad1, grad2 ) -> None:

    print("The Nash equilibrium is: \n", k[ : ])
    #print("The state evolution is: \n", a - k[ 0 ] - k[ 1 ] )
    print("The costs for the related Nash equilibrium are: \n", cost )
    #print("The gradient for the first agent is: \n", grad1)
    #print("The gradient for the second agent is: \n", grad2)

    return None

def perform_sanity_check( a, q, r, k ) -> None:

    k2 = compute_best_response( a, q, r, k[ 0 ] )
    print("Perform sanity check: is the second policies equal to", k2)

    return None

def compute_jacobian( a, q1, q2, r1, r2, k ):

    # initialize the jacobian
    jacobian = np.zeros ( ( 2, 2 ) )

    # first row - derivative of the gradient of the first agent
    jacobian[ 0, 0 ] = compute_second_derivative( a, q1, r1, k[ 0 ], k[ 1 ] )
    jacobian[ 0, 1 ] = compute_mixed_derivative( a, q1, r1, k[ 0 ], k[ 1 ] )

    # second row - derivative of the gradient of the second agent
    jacobian[ 1, 0 ] = compute_mixed_derivative( a, q2, r2, k[ 1 ], k[ 0 ] )
    jacobian[ 1, 1 ] = compute_second_derivative( a, q2, r2, k[ 1 ], k[ 0 ] )

    return jacobian

def is_it_a_saddle( a, q1, q2, r1, r2, k ) -> None:

    # compute the jacobian
    jacobian = compute_jacobian( a, q1, q2, r1, r2, k )

    # compute the eigenvalues of the jacobian
    e_vals, e_vecs = np.linalg.eig( jacobian )

    # print the information obtained
    #print("The Jacobian is: \n", jacobian )
    #print("The eigenvalues are: \n", e_vals )
    if e_vals[ 0 ] * e_vals[ 1 ] < 0:   # if this condtion is respect, the two eigenvalues have different sign
        print("--> This Nash equilibrium is a saddle point <--")
    else:
        print("This Nash equilibrium is NOT a saddle point")

    print("-------------------------------------------------------------------------------------")
    return None


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
        P1 = ARE(A,B1,B2,K1t,K2t,Q1,R1,P1)
        P2 = ARE(A,B2,B1,K2t,K1t,Q2,R2,P2)
        # Rj = np.array([[0.3]])
        # P1 = ARE_crossterms(A,B1,B2,K1t,K2t,Q1,R1,Rj,P1)
        # P2 = ARE_crossterms(A,B2,B1,K2t,K1t,Q2,R2,Rj,P2)

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

# System variable - it is assumed to be positive: a > 0
a = 5

# Cost functions parameters - all of them are assumed to be positive
q1 = 1
q2 = 1
b1 = 1
b2 = 1
r1 = 1 
r2 = 2

# system matrices
A = np.array([[a]])
B1 = np.array([[b1]])
B2 = np.array([[b2]])

# parameters of the cost function of the first agent
Q1 = np.array([[q1]])
R1 = np.array([[r1]])

# parameters of the cost function of the second agent
Q2 = np.array([[q2]])
R2 = np.array([[r2]])

# Time horizon
T = 100



#____________________________________________________________________________________________________________________
#### Compute all the Nash equilibria

# we move to an equivalent system where b1 = b2 = 1
r1 = r1 / ( b1**2 )
r2 = r2 / ( b2**2 )

# possible values of K2
k2 = compute_polynomial_roots( a, q1, q2, r1, r2 )
kNE = np.zeros( ( k2.shape[0], 2 ) )
cost = np.zeros( ( k2.shape[0], 2 ) )
RHGcost = np.zeros( ( k2.shape[0], 3 ) )

print("The roots of the polynomial are:\n", k2)
print("-------------------------------------------------------------------------------------")

nNE = 0 # number of Nash equilibria
# if k2 does not respect the conditions in the if, it can not be a Nash equilibrium
for i in range( k2.shape[0] ):

    if k2[i].imag == 0 and k2[i].real > 0 and k2[i].real < a:

        # take the real part for the second agent
        kNE[ nNE, 1 ] = k2[ i ].real

        # compute the best response for the first agent
        kNE[ nNE, 0 ] = compute_best_response( a, q1, r1, kNE[ nNE, 1 ] )

        # compute the value of the gradients - used for sanity check
        grad1 = compute_gradient( a, q1, r1, kNE[ nNE, 0 ], kNE[ nNE, 1 ] )
        grad2 = compute_gradient( a, q2, r2, kNE[ nNE, 1 ], kNE[ nNE, 0 ] )

        # compute the cost for the two agents
        cost[ nNE, 0 ] = compute_P( a, q1, r1, kNE[ nNE, 0 ], kNE[ nNE, 1 ] )
        cost[ nNE, 1 ] = compute_P( a, q2, r2, kNE[ nNE, 1 ], kNE[ nNE, 0 ] )

        # print the results
        show_Nash_equilibrium( a, kNE[ nNE, : ], cost[ nNE, : ], grad1, grad2 )
        perform_sanity_check( a, q2, r2, kNE[ nNE, : ] )
        is_it_a_saddle( a, q1, q2, r1, r2, kNE[ nNE, : ] )

        RHGcost[ nNE, 0 ] = q1 + r1*(kNE[ nNE, 0 ]**2)
        RHGcost[ nNE, 1 ] = q2 + r2*(kNE[ nNE, 1 ]**2)
        RHGcost[ nNE, 2 ] = RHGcost[ nNE, 0 ] + RHGcost[ nNE, 1 ]

        # update count of Nash equilibria
        nNE += 1

print("The cost for the receding horizon game are:\n", RHGcost)
    
#____________________________________________________________________________________________________________________
#### Divide by b1 and b2 to obtain the original k1 and k2
kNEorig = np.zeros( ( kNE.shape[0], 2 ) )

for i in range( kNE.shape[0] ):
    kNEorig[ i, 0 ]  = kNE[ i, 0 ]/b1
    kNEorig[ i, 1 ]  = kNE[ i, 1 ]/b2

print("The Nash equilibra are: \n", kNEorig[ : ])



#__________________________________________________________
#### Terminal cost of the Nash equilibrium j

j = 1

P1correct = np.array([[ cost[ j, 0] ]])
P2correct = np.array([[ cost[ j, 1] ]])



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

P1 = P1correct + 0.01
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
P2 = P2correct + 0.01

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

# Define colors using the Pastel2 colormap
# Define colors using the Pastel2 colormap
cmap = plt.get_cmap('tab20')
colors = [cmap(i) for i in range(6)] 
cmap2 = plt.get_cmap('Pastel1')
colors2 = [cmap2(i) for i in range(9)] 

# Create figure and axes
fig, axs = plt.subplots(2, 1, figsize=(8, 6))

axs[0].set_facecolor('#FFF5EE')  # Set background color for first subplot
axs[1].set_facecolor('#FFF5EE')  # Set background color for second subplot


# Player 1 plot
axs[0].plot(x, Kfirst[:, 0], '.', markersize=10, label='$k^1$ - First case', color=colors[0])
axs[0].plot(x, Ksecond[:, 0], '.', markersize=10, label='$k^1$ - Second case', color=colors[2])
axs[0].plot(x, Kthird[:, 0], '.', markersize=10, label='$k^1$ - Third case', color=colors[4])
axs[0].plot(x, kNEorig[ 0, 0 ]*np.ones( (T, 1) ), ":", label='$k^{NE,1}$ - First NE', color="black")
axs[0].plot(x, kNEorig[ 1, 0 ]*np.ones( (T, 1) ), "-.", label='$k^{NE,1}$ - Second NE', color="black")
axs[0].plot(x, kNEorig[ 2, 0 ]*np.ones( (T, 1) ), "--", label='$k^{NE,1}$ - Third NE', color="black")
axs[0].set_title('Player 1')
axs[0].set_xlabel('Time step t')
axs[0].set_ylabel('k1')
axs[0].grid(axis="x")
axs[0].set_ylim(0, A)
axs[0].legend(loc='upper left', framealpha=1)

# Player 2 plot
axs[1].plot(x, Kfirst[:, 1], '.', markersize=10, label='$k^2$ - First case', color=colors[0])
axs[1].plot(x, Ksecond[:, 1], '.', markersize=10, label='$k^2$ - Second case', color=colors[2])
axs[1].plot(x, Kthird[:, 1], '.', markersize=10, label='$k^2$ - Third case', color=colors[4])
axs[1].plot(x, kNEorig[ 0, 1 ]*np.ones( (T, 1) ), ":", label='$k^{NE,2}$ - First NE', color="black")
axs[1].plot(x, kNEorig[ 1, 1 ]*np.ones( (T, 1) ), "-.", label='$k^{NE,2}$ - Second NE', color="black")
axs[1].plot(x, kNEorig[ 2, 1 ]*np.ones( (T, 1) ), "--", label='$k^{NE,2}$ - Third NE', color="black")
axs[1].set_title('Player 2')
axs[1].set_xlabel('Time step t')
axs[1].set_ylabel('k2')
axs[1].grid(axis="x")
axs[1].set_ylim(0, A)
axs[1].legend(loc='lower left', framealpha=1)

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