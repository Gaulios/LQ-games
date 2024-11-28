# Giulio Salizzoni @ EPFL
# 2024 May
# Compute the Nash equilibrium of a finite-horizon discrete-time linear quadratic game.
# Then, it take the time horizon to infinite to see what is the equivalent NE.

####################################################################

import numpy as np

# Definisci n
n = 2

# Definisci le matrici A, B1 e B2
A = np.array([[0.588, 0.028],
              [0.570, 0.056]])

B1 = np.array([[1], 
               [1]])

B2 = np.array([[0], 
               [1]])

# Definisci sigma
sigma = 0

# Calcola W
W = sigma**2 * np.eye(2)

# Definisci Q1 e Q2 utilizzando diag
Q1 = np.diag([0.01, 1])
Q2 = np.diag([1, 0.147])

# Definisci R1 e R2
R1 = 0.01
R2 = R1

R1 + R2
1 + 1