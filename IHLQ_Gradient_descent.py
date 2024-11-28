import numpy as np
import matplotlib.pyplot as plt
import control as ct        # used to solve the discrete Lyapunov equation when computing the gradient
import sys                  # used to stop the program in case of issues


"""
Compute the gradient descent in an infinite-horizon discrete-time linear quadratic game.
The game has two agents ( N = 2 ).
The dimension of the state is n = 2. If modified, it is necessary to modify all the matrices accordingly ( A, B1, B2, Q1, Q2, K1, K2 ).
The dimension of the control action is d = 1. It is the same for both agents and can not be modified.

Giulio Salizzoni @ EPFL
2024 July
"""

def compute_gradient( Ag, B, Q, R, K ):

    P = ct.dlyap( Ag, Q + K.T @ R @ K)       # solve the discrete Lyapunov equation
    grad = 2*( ( R + B.T @ P @ B ) @ K - B.T @ P @ Ag )    # compute the gradient

    return grad

def find_step_size( A, B1, B2, K1, K2, grad1, grad2, gamma):
    
    stable = False      # initialize the variable

    while not stable:

        # compute the next step of the gradient
        K1_next = K1 - 2*gamma * grad1
        K2_next = K2 - 2*gamma * grad2

        # check the stability
        stable = check_stability( A - B1 @ K1_next - B2 @ K2_next )

        # decrease gamma
        gamma = 0.5*gamma

    return 2*gamma

def compute_biggest_eigenvalue( M ):

    e_vals, e_vecs = np.linalg.eig( M )            # compute the eigenvalues
    max_e_val = np.max( np.absolute( e_vals) )     # select the biggest eigenvalue in absolute value

    return max_e_val

def check_stability( M ) -> bool:

    max_e_val = compute_biggest_eigenvalue( M )
    
    return max_e_val < 1        # if the condition is true, the system is stable 

def present_results_gradient_descent( variation, epsilon, i, max_number_of_iterations, biggest_e_val, maximum_e_val, K1, K2, grad1, grad2, gamma ) -> None:

    if variation <= epsilon:
        print("The algorithm reached a Nash equilibrium")
        print("-------------------------------------------------------------------------------------")
    elif i == max_number_of_iterations:
        print("The algorithm performed the maximum number of iterations")
        print("-------------------------------------------------------------------------------------")
    elif biggest_e_val >= maximum_e_val:
        print("The algorithm got too close to the unstable region")
        print("-------------------------------------------------------------------------------------")

    print("Last variation was", variation )
    print("The biggest eigenvalue is", biggest_e_val)
    print("The total numer of iterations is", i )
    print("-------------------------------------------------------------------------------------")


    print("The policy reached by agent 1 is: \n", K1[:, i, :] )
    print("The policy reached by agent 2 is: \n", K2[:, i, :] )
    print("The system is equal to: \n", A - B1 @ K1[ :, i, :] - B2 @ K2[ :, i, :])

    print("The last gradient for agent 1 was", grad1)
    print("The last gradient for agent 2 was", grad2)

    show_evolution_policies( K1, K2, gamma, i)

    return None

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

    # Third subplot - gamma
    x = np.linspace( 0, i, i )
    ax3.plot( x, array_type[ 0, 0:i], '.')
    ax3.set_title('Step size')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gamma')
    ax3.grid(True)

    # Show the graphs
    plt.tight_layout()
    plt.show()

    return None


#____________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________

if __name__ == '__main__':
    
    #____________________________________________________________________________________________________________________
    #### Parameters

    #N = 2       # number of agents - fixed
    n = 2        # dimension of the state - can be modified
    #d = 1       # dimension of the agents control action - fixed
    print("-------------------------------------------------------------------------------------")

    # # system matrices
    # A = np.array( [ [0.588, 0.028], [0.570, 0.056] ] )
    # B1 = np.array( [ [1], [1] ] )
    # B2 = np.array( [ [0], [1] ] )

    # # parameters of the cost function of the first agent
    # Q1 = np.array( [ [0.01, 0], [0, 1] ] )
    # R1 = np.array( [ [0.01] ] )

    # # parameters of the cost function of the second agent
    # Q2 = np.array( [ [1, 0], [0, 0.147] ])
    # R2 = np.array( [ [0.01] ] )

    # # initial policies
    # """
    # These policy are close to a Nash equilibrium
    # K1_init = np.array( [ [0.54411873, -0.01564783] ] )
    # K2_init = np.array( [ [0.02210445, 0.07227729] ] )
    # """
    # K1_init = np.array( [ [0.0, -0.01564783] ] )
    # K2_init = np.array( [ [0.02210445, 0.07227729] ] )

    # system matrices
    A = np.array([1.0947])
    B1 = np.array([1])
    B2 = np.array([1])

    # parameters of the cost function of the first agent
    Q1 = np.array( [ 0.11112 ] )
    R1 = np.array( [ 0.40872 / ( 0.10254**2 ) ] )

    # parameters of the cost function of the second agent
    Q2 = np.array( [ 0.25806 ] )
    R2 = np.array( [ 0.5949 / ( 0.045934**2 ) ] )

    # initial policies
    K1_init = np.array( [ 0 ] )
    K2_init = np.array( [-1 ] )


    # check that, given the initial policies, the system is stable
    if not check_stability( A - B1 @ K1_init - B2 @ K2_init):
        sys.exit("The initial policies do not make the system stable \n ---------------------")


    #____________________________________________________________________________________________________________________
    #### Criterium to stop the gradient descent

    # First criterium - it has converged
    epsilon = 0.001              # criterium for convergence
    variation = 1 + epsilon      # initialization of the variable variatioin

    # Second criterium - it has performed too many iterations
    max_number_of_iterations = 5*pow( 10, 5 ) # maximum number of iterations
    i = 0                                   # initialization of the number of iterations

    # Third criterium - it is too close to the unstable region
    maximum_e_val = 0.99
    biggest_e_val = compute_biggest_eigenvalue( A - B1 @ K1_init - B2 @ K2_init )
    

    #____________________________________________________________________________________________________________________
    #### Array to store the values

    gamma_init = 0.00001     # proposed step size for the gradient descent
    gamma = np.zeros( ( 1, max_number_of_iterations ) )

    K1 = np.zeros( ( 1, max_number_of_iterations + 1, n ) )
    K2 = np.zeros( ( 1, max_number_of_iterations + 1, n ) )
    K1[ 0, 0, : ] = K1_init
    K2[ 0, 0, : ] = K2_init


    #____________________________________________________________________________________________________________________
    #### Compute the gradient descent

    while ( variation > epsilon ) and ( i < max_number_of_iterations ) and ( biggest_e_val < maximum_e_val ):

        # compute the gradient for the two players
        grad1 = compute_gradient( A - B2 @ K2[ :, i, :], B1, Q1, R1, K1[ :, i, :] )
        grad2 = compute_gradient( A - B1 @ K1[ :, i, :], B2, Q2, R2, K2[ :, i, :] )

        # find a step size that guarantees the stability of the system (starting from gamma_init)
        gamma[ 0, i] = find_step_size( A, B1, B2, K1[ :, i, :], K2[ :, i, :], grad1, grad2, gamma_init)

        # perform one step
        K1[ 0, i + 1, :] = K1[ 0, i, :] - gamma[ 0, i] * grad1
        K2[ 0, i + 1, :] = K2[ 0, i, :] - gamma[ 0, i] * grad2


        # compute the variation using the frobenius norm of the pseudogradient
        variation = np.linalg.norm( grad1, 'fro' ) + np.linalg.norm( grad2, 'fro' )

        # update the number of iterations
        i = i + 1

        # check how close the system is to the unstable region
        biggest_e_val = compute_biggest_eigenvalue( A - B1 @ K1[ :, i, :] - B2 @ K2[ :, i, :] )


    #____________________________________________________________________________________________________________________
    #### Show the results

    present_results_gradient_descent( variation, epsilon, i, max_number_of_iterations, biggest_e_val, maximum_e_val, K1, K2, grad1, grad2, gamma )
    

    #____________________________________________________________________________________________________________________       
    print("-------------------------------------------------------------------------------------")
    print("Finished")
    print("-------------------------------------------------------------------------------------") 