import numpy as np
import matplotlib.pyplot as plt

"""

"""

# compute the cost for given a certain policy
def computeCost(a,qi,ri,ki,kj):
    Ji = ( qi + ri*( ki**2 ) )/( 1 - ( a - ki - kj )**2 )
    return Ji

# best response in the IH game
def bestResponse(a,qi,ri,kj):
    temp = ri*( 1 - ( a - kj )**2 ) + qi
    br = ( 2*( a - kj)*qi )/( np.sqrt( temp**2 + 4*ri*( ( a - kj)**2 )*qi )  + temp )
    return br

# compute the roots of a polynomial from which we obtain the IH NE
def polynomialRoots(a,q1,q2,r1,r2):

    v1 = a*(r1**2)*(r2**2)
    v2 = -2.5*(a**2)*(r1**2)*(r2**2) + q2*(r1**2)*r2 - q1*r1*(r2**2) + (r1**2)*(r2**2)
    v3 = 2*(a**3)*(r1**2)*(r2**2) - 2*a*q2*(r1**2)*r2 + 2*a*q1*r1*(r2**2) - 2*a*(r1**2)*(r2**2)
    v4 = -0.5*(a**4)*(r1**2)*(r2**2) + (a**2)*q2*(r1**2)*r2 - (a**2)*q1*r1*(r2**2) + (a**2)*(r1**2)*(r2**2) + 0.5*(q2**2)*(r1**2) - 0.5*(q1**2)*(r2**2) - q1*r1*(r2**2) - 0.5*(r1**2)*(r2**2)
    v5 = - a*(q2**2)*(r1**2)
    v6 = 0.5*(a**2)*(q2**2)*(r1**2)

    roots = np.roots([v1,v2,v3,v4,v5,v6])
    return roots

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

# compute the Nash equilibrium
def NE(A,B1,B2,Q1,Q2,R1,R2,P1,P2,T):

    K = np.zeros((T,2))

    for t in range(T-1,-1,-1):

        invM = matrixInverted(B1,B2,R1,R2,P1,P2)

        # Nash equilibrium policies at time t
        Kt = invM @ np.concatenate( ( (B1.T) @ P1 @ A, (B2.T) @ P2 @ A ) , axis = 0)
        K1t = np.array( [Kt[0,:]] )
        K2t = np.array( [Kt[1,:]] )

        # P matrix for the time t-1
        P1 = ARE(A,B1,B2,K1t,K2t,Q1,R1,P1)
        P2 = ARE(A,B2,B1,K2t,K1t,Q2,R2,P2)

        # update the Nash equilibrium policies
        K[t,0] = Kt[0,0]
        K[t,1] = Kt[1,0] 

    return K

# return to which IH NE the FH NE policies at time t = 0 is closer
def indexNEreached(a,k1,k2,kNE,nNE):
    if abs(a - k1 - k2 ) > 1:
        index = -1
    else:
        minimum = abs(k1 - kNE[0,0]) + abs(k2 - kNE[0,1])
        index = 1
        for i in range(1,nNE,1):
            if abs(k1 - kNE[i,0]) + abs(k2 - kNE[i,1]) < minimum:
                index = i + 1
                minimum =  abs(k1 - kNE[i,0]) + abs(k2 - kNE[i,1])
    return index

#____________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________
#### Parameters

# number of agents
n = 2

# system matrices
a = 5
A = np.array([[a]])
b1 = 1
b2 = 1
B1 = np.array([[b1]])
B2 = np.array([[b2]])

# parameters of the cost function of the first agent
q1 = 1
r1= 1
Q1 = np.array([[q1]])
R1 = np.array([[r1]])

# parameters of the cost function of the second agent
q2 = 1
r2 = 2
Q2 = np.array([[q2]])
R2 = np.array([[r2]])

# Time horizon
T = 20


#____________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________
#### Compute the Nash equilibria for the infinite horizon game

# possible values of K2 at Nash equilibrium
k2 = polynomialRoots(a,q1,q2,r1,r2)
kNE = np.zeros((3,2))       # Here we will save the Nash equilibria
cost = np.zeros((3,2))      # Here we will save the costs
nNE = 0                     # Number of Nash equilibria

# verify if the roots are real and compute the corresponding Nash equilibrium
for i in range(k2.shape[0]):

    if k2[i].imag == 0 and k2[i].real > 0 and k2[i].real < a:
        
        # take the real part for the second agent
        kNE[nNE,1] = k2[i].real

        # compute the best response for the first agent
        kNE[nNE,0] = bestResponse(a,q1,r1,kNE[nNE,1])

        # compute the cost for the two agents
        cost[nNE,0] = computeCost(a,q1,r1,kNE[nNE,0],kNE[nNE,1])
        cost[nNE,1] = computeCost(a,q2,r2,kNE[nNE,1],kNE[nNE,0])

        print("#", nNE+1, "The Nash equilibrium is:", kNE[nNE,:])
        print("#", nNE+1, "The state evolution is:", a - kNE[nNE,0] - kNE[nNE,1])
        print("#", nNE+1, "The costs for the related Nash equilibrium are:", cost[nNE,:])
        print("-------------------------------------------------------------------------------------")
        
        #update number of Nash equilibria
        nNE = nNE + 1


#____________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________
#### Compute the Nash equilibria for the finite horizon game with different cost to go

# number of samples
samples = 20

# matrices where to save the results
map = np.zeros((samples,samples))       #here we save to which IH NE the FH NE converged
dist = np.zeros((samples,samples))      #here we save the difference between the IH NE and the policy at time 0 of the FH NE


for i in range(0,samples,1):

    # cost to go for agent 1
    P1 = np.array([[(5*i)/samples + 0.01]])

    for j in range(0,samples,1):

        # cost to go for agent 2
        P2 = np.array([[(5*j)/samples + 0.01]])

        # we compute the FH NE
        K = NE(A,B1,B2,Q1,Q2,R1,R2,P1,P2,T)

        # we check to which NE it is closer
        map[samples - j -1,i] = indexNEreached(a,K[0,0],K[0,1],kNE,nNE)


#____________________________________________________________________________________________________________________
#____________________________________________________________________________________________________________________
#### Plot the map

## plot the convergence map
plt.imshow(map)
plt.colorbar(ticks=[-1, 0, 1, 2, 3], label='Index of the IH NE')
# Titles
plt.title('Convergence of the FH NE to the IH NE')
plt.xlabel('q1')
plt.ylabel('q2')
# corrections to the axis
num_ticks = 5
xticks = np.linspace(0, samples-1, num_ticks)
print(xticks)
yticks = np.linspace(samples-1, 0, num_ticks)
print(yticks)
xlabels = [f'{0.01 + ((5*x)/samples)}' for x in xticks]
#ylabels = [f'{4.76 - ((5*x)/samples)}' for x in ticks]
plt.xticks(xticks,xlabels)
plt.yticks(yticks,xlabels)
# final command
plt.show()




print("-------------------------------------------------------------------------------------")
print("Finished")
print("-------------------------------------------------------------------------------------")