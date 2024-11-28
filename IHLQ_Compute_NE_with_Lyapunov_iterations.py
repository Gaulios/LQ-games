import numpy as np
import matplotlib.pyplot as plt
import sys


"""
Compute a Nash equilibrium of an inifinite-horizon discrete-time linear quadratic game using the Lyapunov iterations.

Giulio Salizzoni @ EPFL
2024 July
"""

def compute_matrix_Pi( A, Bi, Bj, Ki, Kj, Qi, Ri, Pi ):

    # if we consider Pi as the matrix for the cost 
    #Pi_new = Qi + Ki.T @ Ri @ Ki + ( A - Bi @ Ki - Bj @ Kj ).T @ Pi @ ( A - Bi @ Ki - Bj @ Kj )

    # if we use the discrete algebraic Riccati equation
    Ai = A - Bj @ Kj
    Pi_new = Qi + Ai.T @ Pi @ Ai - Ai.T @ Pi @ Bi @ ( np.linalg.inv( Ri + Bi.T @ Pi @ Bi ) ) @ Bi.T @ Pi @ Ai

    return Pi_new

def update_policy_Ki( A, Bi, Bj, Kj, Ri, Pi ):
    
    Ki = np.linalg.inv( Ri + Bi.T @ Pi @ Bi ) @ Bi.T @ Pi @ ( A - Bj @ Kj )

    return Ki

def compute_biggest_eigenvalue( M ):

    e_vals, e_vecs = np.linalg.eig( M )            # compute the eigenvalues
    max_e_val = np.max( np.absolute( e_vals) )     # select the biggest eigenvalue in absolute value

    return max_e_val

def check_stability( M ) -> bool:

    max_e_val = compute_biggest_eigenvalue( M )
    
    return max_e_val < 1        # if the condition is true, the system is stable 

def show_evolution_policies( K1, K2, array_type, i ) -> None:

    # Figure - there are two subplots, one for each agent, put side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    iterations = np.linspace( 0, 10, i )

    # First subplot - agent 1
    sc1 = ax1.scatter( K1[ 0, 0:i, 0 ], K1[ 0, 0:i, 1 ] , c=iterations, cmap='viridis' )
    ax1.set_title('Gradient descent for agent 1')
    ax1.set_xlabel('K1_1')
    ax1.set_ylabel('K1_2')

    # Second subplot - agent 2
    sc2 = ax2.scatter( K2[ 0, 0:i, 0 ], K2[ 0, 0:i, 1 ] , c=iterations, cmap='viridis' )
    ax2.set_title('Gradient descent for agent 2')
    ax2.set_xlabel('K2_1')
    ax2.set_ylabel('K2_2')

    # Third subplot - other array
    x = np.linspace( 0, i, i )
    ax3.plot( x, array_type[ 0 : i ], '.')
    ax3.set_title('Step size')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Variation')
    ax3.grid(True)

    # Show the graphs
    plt.tight_layout()
    plt.show()

    return None


#____________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________

if __name__ == '__main__':
        
    #N = 2       # number of agents
    #n = 2       # dimension of the state
    #d = 1       # dimension of the agents control action
    print("-------------------------------------------------------------------------------------")

    # system matrices
    A = np.array( [ [0.588, 0.028], [0.570, 0.056] ] )
    B1 = np.array( [ [1], [1] ] )
    B2 = np.array( [ [0], [1] ] )

    # parameters of the cost function of the first agent
    Q1 = np.array( [ [0.01, 0], [0, 1] ] )
    R1 = np.array( [ [0.01] ] )

    # parameters of the cost function of the second agent
    Q2 = np.array( [ [1, 0], [0, 0.147] ])
    R2 = np.array( [ [0.01] ] )

    # initial policies
    """""
    The following policies are close to a Nash equilibrium
    K1_init = np.array( [ [0.51459228, 0.04327102] ] )
    K2_init = np.array( [ [0.05691455, 0.0121611] ] )
    """
    K1_init = np.array( [ [0.0, 0.1] ] )
    K2_init = np.array( [ [0.0, 0.0] ] )

    # check that, given the initial policies, the system is stable
    if not check_stability( A - B1 @ K1_init - B2 @ K2_init):
        sys.exit("The initial policies do not make the system stable")


    #____________________________________________________________________________________________________________________
    #### Criteria to stop the algorithm

    # criterium - it has performed too many iterations
    max_number_of_iterations = pow( 10, 3 ) # maximum number of iterations
    i = 0                                   # initialization of the number of iterations

    # criterium - it has converged
    epsilon = 0.0001                 # criterium for convergence
    variation = np.zeros( max_number_of_iterations + 1, dtype = float )
    variation[0] = 1 + epsilon      # initialization of the variable variatioin


    
    #____________________________________________________________________________________________________________________
    #### Compute a Nash equilibrium using Lyapunov iterations

    K1 = np.zeros( ( 1, max_number_of_iterations + 1, 2 ) )
    K2 = np.zeros( ( 1, max_number_of_iterations + 1, 2 ) )
    K1[ 0, 0, : ] = K1_init
    K2[ 0, 0, : ] = K2_init
    P1 = Q1
    P2 = Q2
    
    print("The program reached the while")
    while ( variation[i] > epsilon ) & ( i < max_number_of_iterations ) \
        & ( check_stability( A - B1 @ K1[ :, i, :] - B2 @ K2[ :, i, :] ) ):

        # compute the new values for P1 and P2
        P1_new = compute_matrix_Pi( A, B1, B2, K1[ :, i, :], K2[ :, i, :], Q1, R1, P1 )
        P2_new = compute_matrix_Pi( A, B2, B1, K2[ :, i, :], K1[ :, i, :], Q2, R2, P2 )
                  
        # compute the variation
        variation[ i + 1 ] = np.linalg.norm( P1_new - P1, 'fro' ) + np.linalg.norm( P2_new - P2, 'fro' )
        P1 = P1_new
        P2 = P2_new

        # update the values of K1 and K2
        K1[ :, i + 1, :] = update_policy_Ki( A, B1, B2, K2[ :, i, :], R1, P1 )
        K2[ :, i + 1, :] = update_policy_Ki( A, B2, B1, K1[ :, i, :], R2, P2 )
        
        # update the number of iterations
        i = i + 1


    #____________________________________________________________________________________________________________________
    #### Show the results

    print("Is the final system stable?", check_stability( A - B1 @ K1[ :, i, :] - B2 @ K2[ :, i, :] ) )
    print("The matrix P1 is: \n", P1)
    print("The matrix P2 is: \n", P2)
    print("The control policy K1 is: \n", K1[ :, i - 1, :])
    print("The control policy K2 is: \n", K2[ :, i - 1, :])
    print("Number of iterations:", i)

    show_evolution_policies( K1, K2, variation, i )

    #____________________________________________________________________________________________________________________       
    print("-------------------------------------------------------------------------------------")
    print("Finished")
    print("-------------------------------------------------------------------------------------") 