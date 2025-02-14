import numpy as np
import matplotlib.pyplot as plt
from sympy import *

"""
Computing the Nash equilibria for a scalar two-agents infinite horizon (IH) linear quadratic (LQ) game.
Check if the Nash equilibria are saddle points or not.

System evolution
x_{t+1} = a * x_{t} + u_{1,t} + u_{2,t}

Linear state feedback control
u_{i,t} = k_i x_{t}

Cost function
J_i ( k_1, k_2 ) = sum_{ t = 0 }^\infty ( q_i * ( x_t ** 2 ) + r_i * ( u_{i,t} ** 2 ) )

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

def compute_cost( a, q, r, ki, kj ):

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

    

#____________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________

if __name__ == '__main__':

    print("-------------------------------------------------------------------------------------")
    #____________________________________________________________________________________________________________________
    #### System parameters
    
    # System variable - it is assumed to be positive: a > 0
    a = 5

    # Cost functions parameters - all of them are assumed to be positive
    q1 = 1
    q2 = 2
    b1 = 1
    b2 = 1
    r1 = 1 / ( b1**2 )
    r2 = 1 / ( b2**2 )

    # q1 = 0.11112
    # q2 = 0.25806
    # b1 = 0.10254
    # b2 = 0.045934
    # r1 = 0.40872 / ( b1**2 )
    # r2 = 0.5949 / ( b2**2 )


    #____________________________________________________________________________________________________________________
    #### Compute all the Nash equilibria

    # possible values of K2
    k2 = compute_polynomial_roots( a, q1, q2, r1, r2 )
    kNE = np.zeros( ( k2.shape[0], 2 ) )
    cost = np.zeros( ( k2.shape[0], 2 ) )

    print("The roots of the polynomial are:\n", k2)
    print("-------------------------------------------------------------------------------------")
   
    # if k2 does not respect the conditions in the if, it can not be a Nash equilibrium
    for i in range( k2.shape[0] ):

        if k2[i].imag == 0 and k2[i].real > 0 and k2[i].real < a:

            # take the real part for the second agent
            kNE[ i, 1 ] = k2[ i ].real

            # compute the best response for the first agent
            kNE[ i, 0 ] = compute_best_response( a, q1, r1, kNE[ i, 1 ] )

            # compute the value of the gradients - used for sanity check
            grad1 = compute_gradient( a, q1, r1, kNE[ i, 0 ], kNE[ i, 1 ] )
            grad2 = compute_gradient( a, q2, r2, kNE[ i, 1 ], kNE[ i, 0 ] )

            # compute the cost for the two agents
            cost[ i, 0 ] = compute_cost( a, q1, r1, kNE[ i, 0 ], kNE[ i, 1 ] )
            cost[ i, 1 ] = compute_cost( a, q2, r2, kNE[ i, 1 ], kNE[ i, 0 ] )

            # print the results
            show_Nash_equilibrium( a, kNE[ i, : ], cost[ i, : ], grad1, grad2 )
            perform_sanity_check( a, q2, r2, kNE[ i, : ] )
            is_it_a_saddle( a, q1, q2, r1, r2, kNE[ i, : ] )

        
    #____________________________________________________________________________________________________________________
    #### Divide by b1 and b2 to obtain the original k1 and k2
    kNEorig = np.zeros( ( k2.shape[0], 2 ) )

    for i in range( k2.shape[0] ):

        if k2[i].imag == 0 and k2[i].real > 0 and k2[i].real < a:

            kNEorig[ i, 0 ]  = kNE[ i, 0 ]/b1
            kNEorig[ i, 1 ]  = kNE[ i, 1 ]/b2

    print("The Nash equilibrium is: \n", kNEorig[ : ])







