import numpy as np
import matplotlib.pyplot as plt

"""
Printing gradient descent for a scalar infinite horizon (IH) linear quadratic (LQ) game.
"""

def compute_gradient( a, q, r, ki, kj ):

    grad = ( 2*r*( a - kj )*( ki**2 ) + 2*( r + q - r*( ( a - kj )**2 ) )*ki - 2*( a - kj )*q )/( ( 1 - ( a - ki - kj )**2 )**2 )

    return grad


if __name__ == '__main__':
    
    print("-------------------------------------------------------------------------------------")
    #____________________________________________________________________________________________________________________
    #### Parameters

    # System variable - it is assumed to be positive: a > 0
    a = 1.0947
    b1 = 0.10254
    b2 = 0.045934

    # Cost functions parameters - all of them are assumed to be positive
    q1 = 0.11112
    q2 = 0.25806
    r1 = 0.40872 / ( b1**2 )
    r2 = 0.5949 / ( b2**2 )

    # Parameters for the gradient descent
    eta = 0.001  #step size
    eta1 = eta
    eta2 = eta
    epsilon = 0.01
    max_iteration = 1000

    k1 = 0
    k2 = 0.3
    path = np.array([[k1],[k2]])

     # check that the initial point is stable
    if abs(a - k1 - k2) < 1:
        stable = 1
    else:
        stable = 0

    # initialize parameters
    grad1 = 2*epsilon
    grad2 = 2*epsilon
    i = 0

    while abs(grad1) + abs(grad2) > epsilon and stable == 1 and i < max_iteration:

        i = i + 1
        # recompute the gradient
        grad1 = compute_gradient(a,q1,r1,k1,k2)
        grad2 = compute_gradient(a,q2,r2,k2,k1)

        # gradient descent
        k1 = k1 - eta1*grad1
        k2 = k2 - eta2*grad2

        # check stability
        if abs(a - k1 - k2) >= 1:
            stable = 0

        # save the value
        path = np.concatenate( (path, np.array([[k1],[k2]]) ), axis=1 )


    ################
    #  print the path
    indices = np.arange(len(path[0,:]))
    colormap = plt.cm.inferno
    colors = colormap(indices / max(indices))
    fig, ax = plt.subplots()
    for i in range(len(path[0,:]) - 1):
        ax.plot(path[0,i:i+2], path[1,i:i+2], marker='o', color=colors[i])
    
    # Color bar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=max(indices)))
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Indice')

    # Define the x range
    x = np.linspace(0, a, 1000)

    # Define the lines delimiting the stabelizing policies region
    linear_upper = a + 1 - x  # limit above
    linear_lower = a - 1 - x  # limit below

    plt.plot(x, linear_upper, label='$a + 1 - k_1$', color='black', linestyle='-')
    plt.plot(x, linear_lower, label='$a - 1 - k_1$', color='black', linestyle='-')

    # Titles
    plt.title('Gradient descent path')
    plt.xlabel('K1')
    plt.ylabel('K2')
    plt.xlim(0, a)
    plt.ylim(0, a)

    # Plot
    plt.show()