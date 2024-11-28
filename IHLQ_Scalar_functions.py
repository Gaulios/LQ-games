import numpy as np
import matplotlib.pyplot as plt


def polynomialRoots(a,q1,q2,r1,r2):
    v1 = a*(r1**2)*(r2**2)
    v2 = -2.5*(a**2)*(r1**2)*(r2**2) + q2*(r1**2)*r2 - q1*r1*(r2**2) + (r1**2)*(r2**2)
    v3 = 2*(a**3)*(r1**2)*(r2**2) - 2*a*q2*(r1**2)*r2 + 2*a*q1*r1*(r2**2) - 2*a*(r1**2)*(r2**2)
    v4 = -0.5*(a**4)*(r1**2)*(r2**2) + (a**2)*q2*(r1**2)*r2 - (a**2)*q1*r1*(r2**2) + (a**2)*(r1**2)*(r2**2) + 0.5*(q2**2)*(r1**2) - 0.5*(q1**2)*(r2**2) - q1*r1*(r2**2) - 0.5*(r1**2)*(r2**2)
    v5 = - a*(q2**2)*(r1**2)
    v6 = 0.5*(a**2)*(q2**2)*(r1**2)

    roots = np.roots([v1,v2,v3,v4,v5,v6])
    return roots

def computeCost(a,qi,ri,ki,kj):
    Ji = ( qi + ri*( ki**2 ) )/( 1 - ( a - ki - kj )**2 )
    return Ji

def bestResponse(a,qi,ri,kj):
    temp = ri*( 1 - ( a - kj )**2 ) + qi
    br = ( 2*( a - kj)*qi )/( np.sqrt( temp**2 + 4*ri*( ( a - kj)**2 )*qi )  + temp )
    return br

def gradient(a,qi,ri,ki,kj): 
    grad = ( 2*ri*( a - kj )*( ki**2 ) + 2*( ri + qi - ri*( ( a - kj )**2 ) )*ki - 2*( a - kj )*qi )/( ( 1 - ( a - ki - kj )**2 )**2 )
    return grad

def gradientDescent(a,q1,r1,q2,r2,k1,k2,eta1,eta2,epsilon):
    
    grad1 = gradient(a,q1,r1,k1,k2)
    grad2 = gradient(a,q2,r2,k2,k1)
    if abs(a - k1 - k2) < 1:
        stable = 1
    else:
        stbale = 0

    while abs(grad1) + abs(grad2) > epsilon  and stable == 1:
        # gradient descent
        k1 = k1 - eta1*grad1
        k2 = k2 - eta2*grad2
        # check stability
        if abs(a - k1 - k2) > 0.99:
            stable = 0
            k1 = -10
            k2 = -10
        # recompute the gradient
        grad1 = gradient(a,q1,r1,k1,k2)
        grad2 = gradient(a,q2,r2,k2,k1)

    return np.array([[k1],[k2]])

def pathGradientDescent(a,q1,r1,q2,r2,k1,k2,eta1,eta2,epsilon):
    
    grad1 = gradient(a,q1,r1,k1,k2)
    grad2 = gradient(a,q2,r2,k2,k1)
    path = np.array([[k1],[k2]])
    if abs(a - k1 - k2) < 1:
        stable = 1
    else:
        stable = 0

    while abs(grad1) + abs(grad2) > epsilon and stable == 1:
        # gradient descent
        k1 = k1 - eta1*grad1
        k2 = k2 - eta2*grad2
        # check stability
        if abs(a - k1 - k2) >= 1:
            stable = 0
        # save the value
        path = np.concatenate( (path, np.array([[k1],[k2]]) ), axis=1 )
        # recompute the gradient
        grad1 = gradient(a,q1,r1,k1,k2)
        grad2 = gradient(a,q2,r2,k2,k1)

    return path, stable

def indexNEreached(a,k1,k2,kNE,nNE):
    if abs(a - k1 - k2 ) > 1:
        index = -2
    else:
        minimum = abs(k1 - kNE[0,0]) + abs(k2 - kNE[0,1])
        index = 1
        for i in range(1,nNE,1):
            if abs(k1 - kNE[i,0]) + abs(k2 - kNE[i,1]) < minimum:
                index = i + 1
                minimum =  abs(k1 - kNE[i,0]) + abs(k2 - kNE[i,1])
    return index
