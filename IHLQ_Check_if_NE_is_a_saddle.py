import numpy as np
import matplotlib.pyplot as plt
import control as ct        # used to solve the discrete Lyapunov equation when computing the gradient
import sys                  # used to stop the program in case of issues



"""
Check whether a Nash equilibrium is a saddle point or not.

Giulio Salizzoni @ EPFL
2024 July
"""


def compute_gradient( Ag, B, Q, R, K ):

    P = ct.dlyap( Ag, Q + K.T @ R @ K)       # solve the discrete Lyapunov equation
    grad = 2*( ( R + B.T @ P @ B ) @ K - B.T @ P @ Ag )    # compute the gradient

    return grad

def compute_biggest_eigenvalue( M ):

    e_vals, e_vecs = np.linalg.eig( M )            # compute the eigenvalues
    max_e_val = np.max( np.absolute( e_vals) )     # select the biggest eigenvalue in absolute value

    return max_e_val

def check_stability( M ) -> bool:

    max_e_val = compute_biggest_eigenvalue( M )
    
    return max_e_val < 1        # if the condition is true, the system is stable 

def compute_derivative_of_gradient( A, Bi, Bj, Qi, Ri, Ki, Kj, epsilon ):


    partial_jacobian = np.zeros( ( 2, 4 ) )

    diff = np.zeros( ( 1, 2 ) )
    diff[ 0, 0] = epsilon

    A_plus = A - Bj @ ( Kj + diff )
    A_minus = A - Bj @ ( Kj - diff )

    grad_plus_eps = compute_gradient( A_plus, Bi, Qi, Ri, Ki )
    grad_minus_eps = compute_gradient( A_minus, Bi, Qi, Ri, Ki )

    partial_jacobian[ :, 0 : 2 ] = ( grad_plus_eps - grad_minus_eps ) / ( 2 * epsilon )


    diff = np.zeros( ( 1, 2 ) )
    diff[ 0, 1] = epsilon

    A_plus = A - Bj @ ( Kj + diff )
    A_minus = A - Bj @ ( Kj - diff )

    grad_plus_eps = compute_gradient( A_plus, Bi, Qi, Ri, Ki )
    grad_minus_eps = compute_gradient( A_minus, Bi, Qi, Ri, Ki )

    partial_jacobian[ :, 2 : 4 ] = ( grad_plus_eps - grad_minus_eps ) / ( 2 * epsilon )

    return partial_jacobian


#____________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________

if __name__ == '__main__':
        
    N = 2       # number of agents - fixed
    n = 2       # dimension of the state - can be modified
    d = 1       # dimension of the agents control action - fixed
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
    """
    K1 = np.array( [ [0.54411873, -0.01564783] ] )
    K2 = np.array( [ [0.02210445, 0.07227729] ] )
    """
    K1 = np.array( [ [0.54411873, -0.01564783] ] )
    K2 = np.array( [ [0.02210445, 0.07227729] ] )

    # check that, given the initial policies, the system is stable
    if not check_stability( A - B1 @ K1 - B2 @ K2 ):
        sys.exit("The initial policies do not make the system stable \n ---------------------")

    
    #____________________________________________________________________________________________________________________
    #### Checking a Nash equilibrium

    """
    We want to compute the Jacobian of the pseudogradient.

    The dimension of the pseudogradient ( vectorized ) is the product of:
    - the number of agents N;
    - the dimension of the control matrices, which are given by dimension of the state n times the dimension of the control action d.
    Thus, in our case, the pseudogradients has N * n * d terms.

    The Jacobian will have ( N * n * d )^2 terms.
    """

    epsilon = 0.01

    # Jacobian
    jacobian = np.zeros( ( 1, N * n * d, N * n * d))

    print("The initialization of the Jacobian is: \n", jacobian )

    ## agent 1
    agent = 0

    initial_index = agent * ( n * d )
    final_index = ( agent + 1 ) * ( n * d )

    #jacobian[ :, initial_index : final_index , : ] = compute_derivative_of_gradient( A, B1, B2, Q1, R1, K1, K2, epsilon )
    # First term --------------------------
    diff = np.zeros( ( 1, 2 ) )
    diff[ 0, 0] = epsilon

    A1 = A - B2 @ K2

    grad_plus_eps = compute_gradient( A1, B1, Q1, R1, K1 + diff )
    grad_minus_eps = compute_gradient( A1, B1, Q1, R1, K1 - diff )


    jacobian[ :, initial_index : final_index , 0 ] = ( grad_plus_eps - grad_minus_eps ) / ( 2 * epsilon )

    # Second term --------------------------
    diff = np.zeros( ( 1, 2 ) )
    diff[ 0, 1] = epsilon

    A1 = A - B2 @ K2

    grad_plus_eps = compute_gradient( A1, B1, Q1, R1, K1 + diff )
    grad_minus_eps = compute_gradient( A1, B1, Q1, R1, K1 - diff )

    jacobian[ :, initial_index : final_index , 1 ] = ( grad_plus_eps - grad_minus_eps ) / ( 2 * epsilon )

    # Third term --------------------------
    diff = np.zeros( ( 1, 2 ) )
    diff[ 0, 0] = epsilon

    A_plus = A - B2 @ ( K2 + diff )
    A_minus = A - B2 @ ( K2 - diff )

    grad_plus_eps = compute_gradient( A_plus, B1, Q1, R1, K1 )
    grad_minus_eps = compute_gradient( A_minus, B1, Q1, R1, K1 )

    jacobian[ :, initial_index : final_index , 2 ] = ( grad_plus_eps - grad_minus_eps ) / ( 2 * epsilon )

    # Fourth term --------------------------
    diff = np.zeros( ( 1, 2 ) )
    diff[ 0, 1] = epsilon

    A_plus = A - B2 @ ( K2 + diff )
    A_minus = A - B2 @ ( K2 - diff )

    grad_plus_eps = compute_gradient( A_plus, B1, Q1, R1, K1 )
    grad_minus_eps = compute_gradient( A_minus, B1, Q1, R1, K1 )

    jacobian[ :, initial_index : final_index , 3 ] = ( grad_plus_eps - grad_minus_eps ) / ( 2 * epsilon )
    print("-------------------------------------------------------------------------------------")


    print("-------------------------------------------------------------------------------------")

    ## agent 2
    agent = 1

    initial_index = agent * ( n * d )
    final_index = ( agent + 1 ) * ( n * d )

    #jacobian[ :, initial_index : final_index , : ] = compute_derivative_of_gradient( A, B2, B1, Q2, R2, K2, K1, epsilon )

    # First term --------------------------
    diff = np.zeros( ( 1, 2 ) )
    diff[ 0, 0] = epsilon

    A2 = A - B1 @ K1

    grad_plus_eps = compute_gradient( A2, B2, Q2, R2, K2 + diff )
    grad_minus_eps = compute_gradient( A2, B2, Q2, R2, K2 - diff )


    jacobian[ :, initial_index : final_index , 0 ] = ( grad_plus_eps - grad_minus_eps ) / ( 2 * epsilon )

    # Second term --------------------------
    diff = np.zeros( ( 1, 2 ) )
    diff[ 0, 1] = epsilon

    A2 = A - B1 @ K1

    grad_plus_eps = compute_gradient( A2, B2, Q2, R2, K2 + diff )
    grad_minus_eps = compute_gradient( A2, B2, Q2, R2, K2 - diff )

    jacobian[ :, initial_index : final_index , 1 ] = ( grad_plus_eps - grad_minus_eps ) / ( 2 * epsilon )

    # Third term --------------------------
    diff = np.zeros( ( 1, 2 ) )
    diff[ 0, 0] = epsilon

    A_plus = A - B1 @ ( K1 + diff )
    A_minus = A - B1 @ ( K1 - diff )

    grad_plus_eps = compute_gradient( A_plus, B2, Q2, R2, K2 )
    grad_minus_eps = compute_gradient( A_minus, B2, Q2, R2, K2 )

    jacobian[ :, initial_index : final_index , 2 ] = ( grad_plus_eps - grad_minus_eps ) / ( 2 * epsilon )

    # Fourth term --------------------------
    diff = np.zeros( ( 1, 2 ) )
    diff[ 0, 1] = epsilon

    A_plus = A - B1 @ ( K1 + diff )
    A_minus = A - B1 @ ( K1 - diff )

    grad_plus_eps = compute_gradient( A_plus, B2, Q2, R2, K2 )
    grad_minus_eps = compute_gradient( A_minus, B2, Q2, R2, K2 )

    jacobian[ :, initial_index : final_index , 3 ] = ( grad_plus_eps - grad_minus_eps ) / ( 2 * epsilon )
    print("-------------------------------------------------------------------------------------")


    ## compute eigenvalues of the Jacobian

    e_vals, e_vecs = np.linalg.eig( jacobian )

    
    #____________________________________________________________________________________________________________________
    #### Show the results

    print("The Jacobian is: \n", jacobian)
    print("The eigenvalues are: \n", e_vals)
    


    
        


