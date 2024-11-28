import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, Array

"""
Computing the Nash equilibria for a scalar infinite horizon (IH) linear quadratic (LQ) game.
"""



a, ki, kj, q, r = symbols("a ki kj q r")

costo = ( q + r * ( ki**2 ) ) / ( 1 - ( a - ki - kj )**2 )
print( "The cost function is: \n", costo )
print("-------------------------------------------------------------------------------------")

grad = diff( costo, ki )
print( "The gradient is: \n", grad )
print("-------------------------------------------------------------------------------------")

first = diff( grad, ki )
print( "The second derivative with respect to k1 is: \n", first )
print("-------------------------------------------------------------------------------------")

second = diff( grad, kj )
print( "The mixed derivative is: \n", second )
print("-------------------------------------------------------------------------------------")

print( first.subs( [ ( a, 5 ), ( q, 1 ), ( r, 1 ), ( ki, 2.5 ), ( kj, 2 ) ] ) )
