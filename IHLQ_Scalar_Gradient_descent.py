import numpy as np
import matplotlib.pyplot as plt

"""
Computing the Nash equilibria for a scalar infinite horizon (IH) linear quadratic (LQ) game.
"""

def compute_polynomial_roots( a, q1, q2, r1, r2 ):
    v1 = a*(r1**2)*(r2**2)
    v2 = -2.5*(a**2)*(r1**2)*(r2**2) + q2*(r1**2)*r2 - q1*r1*(r2**2) + (r1**2)*(r2**2)
    v3 = 2*(a**3)*(r1**2)*(r2**2) - 2*a*q2*(r1**2)*r2 + 2*a*q1*r1*(r2**2) - 2*a*(r1**2)*(r2**2)
    v4 = -0.5*(a**4)*(r1**2)*(r2**2) + (a**2)*q2*(r1**2)*r2 - (a**2)*q1*r1*(r2**2) + (a**2)*(r1**2)*(r2**2) + 0.5*(q2**2)*(r1**2) - 0.5*(q1**2)*(r2**2) - q1*r1*(r2**2) - 0.5*(r1**2)*(r2**2)
    v5 = - a*(q2**2)*(r1**2)
    v6 = 0.5*(a**2)*(q2**2)*(r1**2)

    roots = np.roots([v1,v2,v3,v4,v5,v6])
    return roots

def compute_cost( a, q, r, ki, kj ):

    cost = ( q + r*( ki**2 ) )/( 1 - ( a - ki - kj )**2 )

    return cost

def compute_best_response( a, q, r, kj ):

    temp = r * ( 1 - ( a - kj )**2 ) + q
    best_response = ( 2 * ( a - kj) * q ) / ( np.sqrt( temp**2 + 4 * r * ( ( a - kj)**2 ) * q )  + temp )

    return best_response

def compute_inverse_best_response( a, q, r, kj ):

    ki = (q - ( ( kj**4 )*( r**2 ) + 2*( kj**2 )*q*r + 4*( kj**2 )*( r**2 ) + ( q**2 ) )**(1/2) - ( kj**2 )*r + 2*a*kj*r )/( 2*kj*r )

    return ki

def compute_gradient( a, q, r, ki, kj ):

    grad = ( 2*r*( a - kj )*( ki**2 ) + 2*( r + q - r*( ( a - kj )**2 ) )*ki - 2*( a - kj )*q )/( ( 1 - ( a - ki - kj )**2 )**2 )

    return grad

def perform_gradient_descent( a, q1, r1, q2, r2, k1, k2, eta1, eta2, epsilon ):

    # check that the initial point is stable
    if abs(a - k1 - k2) < 1:
        stable = 1
    else:
        stable = 0
        k1 = -10
        k2 = -10
        return np.array([[k1],[k2]])

    # compute the gradient
    grad1 = compute_gradient( a, q1, r1, k1, k2 )
    grad2 = compute_gradient( a, q2, r2, k2, k1 )

    # perform the gradient descent
    while abs(grad1) + abs(grad2) > epsilon  and stable == 1:

        # gradient descent
        k1 = k1 - eta1*grad1
        k2 = k2 - eta2*grad2

        stable = 0

        # check stability
        if abs(a - k1 - k2) < 1:

            stable = 1
            # compute the gradient
            grad1 = compute_gradient( a, q1, r1, k1, k2 )
            grad2 = compute_gradient( a, q2, r2, k2, k1 )

    return np.array([[k1],[k2]])

def path_gradient_descent( a, q1, r1, q2, r2, k1, k2, eta1, eta2, epsilon ):

    # check that the initial point is stable
    if abs(a - k1 - k2) < 1:
        stable = 1
    else:
        stable = 0
    
    grad1 = 2*epsilon #value used to the enter the while
    grad2 = 0

    path = np.array([[k1],[k2]])

    while abs(grad1) + abs(grad2) > epsilon and stable == 1:

        # compute the gradient
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

    return path, stable

# Important
def determine_index_NE_reached( a, k1, k2, kNE, nNE ):

    # check that the system is stable
    if abs(a - k1 - k2 ) > 1:
        index = -2
        return index

    # compute the distance from the first Nash equilibrium
    minimum = abs(k1 - kNE[0,0]) + abs(k2 - kNE[0,1])

    # associate the index to the fist Nash equilibrium
    index = 1

    # check if the policies are closer to another Nash equilibrium
    for i in range(1,nNE):

        if abs(k1 - kNE[i,0]) + abs(k2 - kNE[i,1]) < minimum:

            # update the index and the distance
            index = i + 1
            minimum =  abs(k1 - kNE[i,0]) + abs(k2 - kNE[i,1])

    return index
# ______________________________________________________

def show_values_NE( nNE, kNE, cost ) -> None:

    print("#", nNE+1, "The Nash equilibrium is:", kNE[nNE,:])
    print("#", nNE+1, "The state evolution is:", a - kNE[nNE,0] - kNE[nNE,1])
    print("#", nNE+1, "The costs for the related Nash equilibrium are:", cost[nNE,:])
    print("-------------------------------------------------------------------------------------")
            
    return None

# Important
def show_convergence_map(a, b1, b2, q1, r1, q2, r2, kMin, kMax, nSamples, map, kNE):
    
    br1 = np.zeros(nSamples + 2)
    ibr2 = np.zeros(nSamples + 2)
    x = np.linspace(kMin, kMax, nSamples + 2)

    # Compute the best response and inverse best response
    for i in range(nSamples + 2):
        br1[i] = (nSamples - 1) * (kMax - compute_best_response(a, q2, r2, x[i])) / (kMax - kMin)
        if x[i] == 0:
            ibr2[i] = (nSamples - 1) * (kMax - a) / (kMax - kMin)
        else:
            ibr2[i] = (nSamples - 1) * (kMax - compute_inverse_best_response(a, q1, r1, x[i])) / (kMax - kMin)

    # Define custom colormap
    custom_cmap = get_colors_for_covergence_map()

    # Plot the convergence map
    plt.imshow(map, cmap=custom_cmap, origin="upper", interpolation='none')

    # Adjust x and y labels to show correct k1 and k2 values
    labels, locations = plt.yticks()
    newLabelsx = labelx_convergence_map(b1, kMin, kMax, labels)
    newLabelsy = labely_convergence_map(b2, kMin, kMax, labels)
    plt.xticks(labels, newLabelsx)
    plt.yticks(labels, newLabelsy)

    # Ensure the axis is set properly
    plt.xlim(0, nSamples - 1)
    plt.ylim(nSamples - 1, 0)  # Flip to match imshow orientation

    # Convert Nash Equilibria to correct pixel coordinates
    position_NE = np.zeros_like(kNE)
    position_NE[:, 0] = (nSamples - 1) * (kNE[:, 0] - kMin) / (kMax - kMin)  # k1 to x-axis
    position_NE[:, 1] = (nSamples - 1) * (kMax - kNE[:, 1]) / (kMax - kMin)  # k2 to y-axis

    # Plot the Nash Equilibria
    plt.scatter(position_NE[:, 0], position_NE[:, 1], color='purple', marker='*', s=200, label='NE')

    # Plot the best response functions
    xplot = np.linspace(-1, nSamples + 1, nSamples + 2)
    plt.plot(xplot, br1, label='$br_2(k_1)$', color='blue')
    plt.plot(xplot, ibr2, label='$br_1^{-1}(k_1)$', color='red')

    # Define k1 values for the line
    k1_values = np.linspace(kMin, kMax, 100)

    # Compute k2 values for the two boundary conditions
    k2_line1 = a - 1 - k1_values
    k2_line2 = a + 1 - k1_values

    # Convert only the valid points to pixel coordinates
    k1_pixel1 = (nSamples - 1) * (k1_values - kMin) / (kMax - kMin)
    k2_pixel1 = (nSamples - 1) * (kMax - k2_line1) / (kMax - kMin)

    k1_pixel2 = (nSamples - 1) * (k1_values - kMin) / (kMax - kMin)
    k2_pixel2 = (nSamples - 1) * (kMax - k2_line2) / (kMax - kMin)

    # Plot the valid parts of the lines
    plt.plot(k1_pixel1, k2_pixel1, color='black', linestyle='-', label='$|a - b_1 k_1 - b_2 k_2| = 1$')
    plt.plot(k1_pixel2, k2_pixel2, color='black', linestyle='-')


    # Labels, legend, and title
    plt.legend()
    plt.xlabel('$k_1$', fontsize=16)
    plt.ylabel('$k_2$', fontsize=16)
    plt.show()

    return None

# ______________________________________________________

    
def get_colors_for_covergence_map():

    # Get the Pastel1 colormap
    color_map = plt.get_cmap('Pastel1')

    # Initialize a list to store 5 colors (as tuples)
    colors = []  

    # Assign specific colors from the colormap
    colors.append(color_map(0))       # First color from the colormap
    colors.append(color_map(8 / 8))   # Color corresponding to 1/8
    colors.append(color_map(4 / 8))   # Color corresponding to 2/8
    colors.append(color_map(2 / 8))   # Color corresponding to 4/8
    colors.append(color_map(5 / 8))   # Last color from the colormap

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(colors)

    return custom_cmap

def labelx_convergence_map( b1, kMin, kMax, labels ):
    
    labelx = np.round( ( ( (kMax - kMin )*labels )/max(labels) + kMin*np.ones( np.shape(labels) ) )/b1 , 3 )
    return labelx

def labely_convergence_map( b2, kMin, kMax, labels ):
    
    invLabels = labels[::-1]
    labely = np.round( ( ( (kMax - kMin )*invLabels )/max(labels) + ( (kMax - kMin ) / 5 )*np.ones( np.shape(labels) ) )/b2 , 3 )
    return labely

# Important
def show_flow_map( b1, b2,  kMin, kMax, flow_map_x, flow_map_y, kNE ) -> None:

    # Get the dimensions of the grid
    rows, cols = flow_map_x.shape

    # Create the grid
    x = np.linspace(0, cols-1, cols)
    y = np.linspace(0, rows-1, rows)
    X, Y = np.meshgrid(x, y)

    # Create the streamplot
    plt.figure(figsize=(8, 8))
    plt.streamplot( X, Y, flow_map_x, flow_map_y, density = 3, zorder=1 )
    plt.title('Flow of the gradient')
    plt.xlabel('k1')
    plt.ylabel('k2')

    # add the scale on the x and y axis
    labels, locations = plt.yticks()
    newLabelsx = labelx_convergence_map( b1, kMin, kMax, labels )
    newLabelsy = labelx_convergence_map( b2, kMin, kMax, labels )
    plt.xticks( labels, newLabelsx )
    plt.yticks(labels, newLabelsy )


    # Plot a star at the Nash equilibria
    position_NE = find_position_NE_flow( kMin, kMax, nSamples, kNE)
    plt.scatter( position_NE[ : , 0 ] , position_NE[ : , 1 ], color='purple', marker='*', s=200, label='NE')

    # show plot
    plt.show()

    return None
# ______________________________________________________

def find_position_NE_flow( kMin, kMax, nSamples, kNE ):

    deltak = (kMax - kMin)/nSamples

    # Get the number of Nash equilibria
    rows, cols = kNE.shape

    coordinate = np.zeros( (rows, 2 ) )

    for i in range( 0, rows ):
        coordinate[ i, 0 ] = np.round( ( kNE[ i, 0 ] - kMin ) / deltak )
        coordinate[ i, 1 ] = np.round( nSamples - ( kMax - kNE[ i, 1 ] ) / deltak + 2 )

    return coordinate

#____________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________

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
    eta = 0.0004  #step size
    eta1 = eta
    eta2 = eta
    epsilon = 0.01

    #____________________________________________________________________________________________________________________
    #### Compute the Nash equilibria

    kNE = np.zeros((3,2))       # Here we will save the Nash equilibria
    cost = np.zeros((3,2))      # Here we will save the costs
    nNE = 0                     # Number of Nash equilibria

    # possible values of K2 at Nash equilibrium
    k2 = compute_polynomial_roots(a,q1,q2,r1,r2)

    # if k2 does not respect the conditions in the if, it can not be a Nash equilibrium
    for i in range(k2.shape[0]):

        if k2[i].imag == 0 and k2[i].real > 0 and k2[i].real < a:
            
            # take the real part for the second agent (the imaginary part is zero)
            kNE[nNE,1] = k2[i].real

            # compute the best response for the first agent
            kNE[nNE,0] = compute_best_response(a,q1,r1,kNE[nNE,1])

            # compute the cost for the two agents
            cost[nNE,0] = compute_cost(a,q1,r1,kNE[nNE,0],kNE[nNE,1])
            cost[nNE,1] = compute_cost(a,q2,r2,kNE[nNE,1],kNE[nNE,0])

            # print the values
            show_values_NE( nNE, kNE, cost )

            #update number of Nash equilibria
            nNE = nNE + 1

    kNE = kNE[0:nNE,:]
    cost = cost[0:nNE,:]

    #____________________________________________________________________________________________________________________
    #### Compute the convergence map and the flow map

    nSamples = 300
    convergence_map = np.zeros( ( nSamples, nSamples ) )
    flow_map_x = np.zeros( ( nSamples, nSamples ) )
    flow_map_y = np.zeros( ( nSamples, nSamples ) )

    # initial and final values for the policies
    kMin = 0
    kMax = 0.2
    # variation
    deltak = (kMax - kMin)/nSamples

    for i in range(0,nSamples,1):

        k2 = kMax - ( i + 1) * deltak   # k2 will be on the y axis
        print(".")
        for j in range(0,nSamples,1):
            
            # starting point
            k1 = kMin + j * deltak          # k1 will be on the x axis

            # the initial policies are stabilizing
            if abs( a - k1 - k2 ) < 1:

                # compute all the gradient descent
                kConv = perform_gradient_descent( a, q1, r1, q2, r2, k1, k2, eta1, eta2, epsilon )

                # check to which Nash equilibrium the gradient descent converged
                index = determine_index_NE_reached( a, kConv[0,0], kConv[1,0], kNE, nNE )

                # save it in the matrix
                convergence_map[ i, j ] = index

                # We will plot the flow using streamplot. For streamplot, the value (0,0) is in the left top corner.
                flow_map_x[ nSamples - i - 1, j ] = -compute_gradient( a, q1, r1, k1, k2 )
                flow_map_y[ nSamples - i - 1, j ] = -compute_gradient( a, q2, r2, k2, k1 )

            # the initial policies are not stabilizing
            else: 
                convergence_map[ i, j ] = -1
                flow_map_x[ nSamples - i - 1, j ] = 0
                flow_map_y[ nSamples - i - 1, j ] = 0

        
    #plot the convergence map
    show_convergence_map( a, b1, b2, q1, r1, q2, r2, kMin, kMax, nSamples, convergence_map, kNE )

    #plot the flow map
    #show_flow_map( b1, b2,  kMin, kMax, flow_map_x, flow_map_y, kNE)


    #____________________________________________________________________________________________________________________
    #### 
    
    # k1 = kNE[1,0] + 0.001
    # k2 = kNE[1,1] + 0.001
    # (path,stable) = path_gradient_descent( a, q1, r1, q2, r2, k1, k2, 0.0004, 0.0004, epsilon )
    # # Did the algorithm converged to a stable solution?
    # print("The final point reached is stable?\n", stable)
    # # To which Nash equilibrium did it converge?
    # l = path[0,:].size
    # ind = determine_index_NE_reached(a,path[0,l-1],path[1,l-1],kNE,nNE)
    # print("The NE reached is number:", ind)
    # print(path[0,l-1],path[1,l-1])


    # indices = np.arange(len(path[0,:]))
    # colormap = plt.cm.inferno
    # colors = colormap(indices / max(indices))
    # fig, ax = plt.subplots()
    # for i in range(len(path[0,:]) - 1):
    #     ax.plot(path[0,i:i+2], path[1,i:i+2], marker='o', color=colors[i])
    # # Color bar
    # sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0, vmax=max(indices)))
    # cbar = fig.colorbar(sm, ax=ax)
    # cbar.set_label('Indice')
    # # Titles
    # plt.title('Gradient descent path')
    # plt.xlabel('K1')
    # plt.ylabel('K2')
    # plt.xlim(0, a)
    # plt.ylim(0, a)
    # # Plot
    # plt.show()


    #____________________________________________________________________________________________________________________
    print("-------------------------------------------------------------------------------------")
    print("Finished")
    print("-------------------------------------------------------------------------------------")