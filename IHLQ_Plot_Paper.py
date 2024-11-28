import numpy as np
import matplotlib.pyplot as plt

def compute_best_response( a, q, r, kj ):

    temp = r*( 1 - ( a - kj )**2 ) + q
    ki = ( 2*( a - kj )*q )/( np.sqrt( temp**2 + 4*r*( ( a - kj )**2 )*q )  + temp )

    return ki

def compute_inverse_best_response( a, q, r, kj ):

    ki = (q - ( ( kj**4 )*( r**2 ) + 2*( kj**2 )*q*r + 4*( kj**2 )*( r**2 ) + ( q**2 ) )**(1/2) - ( kj**2 )*r + 2*a*kj*r )/( 2*kj*r )

    return ki

#____________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________

if __name__ == '__main__':
    print("-------------------------------------------------------------------------------------")
    #____________________________________________________________________________________________________________________
    #### System parameters
    
    # System variable - it is assumed to be positive: a > 0
    a = 1.5

    # Cost functions parameters - all of them are assumed to be positive
    q1 = 1
    q2 = 1
    b1 = 1
    b2 = 1
    r1 = 3 / ( b1**2 )
    r2 = 5 / ( b2**2 )


    # compute the best response function br{2}(k_1) and
    # the inverse best response function br{1}^{-1}(k_1)
    samples = 1000
    br2 = np.zeros( samples )
    ibr1 = np.zeros( samples ) 

    # Define the x range
    x = np.linspace(0, a, samples)

    # Compute the best response of agent 2 and the inverse best response of agent 1
    for i in range( samples ) :
        
        br2[i] = compute_best_response( a, q2, r2, x[i] )
        if x[i] == 0: # the inverse best response is not well defined in kj = 0
            ibr1[i] = a
        else:
            ibr1[i] = compute_inverse_best_response( a, q1, r1, x[i] )


    # Define the lines delimiting the stabelizing policies region
    linear_upper = a + 1 - x  # limit above
    linear_lower = a - 1 - x  # limit below


    # Create the plot
    plt.figure(figsize=(8, 8))

    # Get colormap
    color_map = plt.get_cmap('Pastel1')


    # Plot the functions
    plt.plot(x, br2, label='$br_2(k_1)$', color='blue')
    plt.plot(x, ibr1, label='$br_1^{-1}(k_1)$', color='red')
    plt.plot(x, linear_upper, label='$a + 1 - k_1$', color='black', linestyle='-')
    plt.plot(x, linear_lower, label='$a - 1 - k_1$', color='black', linestyle='-')

    # Fill the areas between br2, ibr1 and the stabilizing region limits
    plt.fill_between(x, linear_upper, br2, color=color_map(2), alpha=1)
    plt.fill_between(x, ibr1, linear_lower, color=color_map(5), alpha=1)

    # Fill the areas between br2 and ibr1
    plt.fill_between(x, br2, ibr1, where=(br2 > ibr1), color=color_map(0), alpha=1)
    plt.fill_between(x, br2, ibr1, where=(br2 < ibr1), color=color_map(4), alpha=1)

    # Fill in grey the unstable region
    plt.fill_between(x, linear_upper, np.max(linear_upper), color=color_map(6), alpha=0.3)
    plt.fill_between(x, linear_lower, np.min(linear_lower), color=color_map(6), alpha=0.3)

    # Find the Nash equilibria
    NE = np.isclose(br2, ibr1, atol=1e-3)

    # Get the x and y coordinates of the Nash equilibria
    x_NE = x[NE]
    y_NE = br2[NE]  # or ibr1[NE] since ibr1 = br2 at these points

    # Plot a star at the Nash equilibria
    plt.scatter(x_NE, y_NE, color='purple', marker='*', s=200, label='NE')

    # Define the region to plot
    plt.xlim(0, a )
    plt.ylim(0, a )

    # Add labels and legends
    plt.xlabel('$k_1$', fontsize=20)
    plt.ylabel('$k_2$', fontsize=20)
    plt.legend(fontsize=16)

    # Display the plot
    plt.show()