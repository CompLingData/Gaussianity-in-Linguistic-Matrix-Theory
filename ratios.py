# File Purpose: Extracts the experimental expectation values of linear, quadratic, cubic and quartic graphs, calculates
# the theoretical expectation value equivalent and computes the ratio. It also confirms that the convergence criteria is
# met.

from __future__ import division
import numpy as np
import shelve
import sys
from scipy.optimize import fsolve

# Allowing user inputs: Dimension and File name.
while True:
    try:
        D = input("Enter the dimension of the matrices: ")
        D = int(D)
        break
    except ValueError:
        print("Please select an integer value for the dimension")

fname = input("Enter the name of the shelve file: ")

# Opening the matrix data shelve to read in the data
ts = shelve.open(fname, flag='r', protocol=2)
dic_key = ts.keys()
M = np.zeros((23, len(dic_key)))

# Iterating over all matrices, and calculating the values of the matrix function sums
for widx, k in enumerate(dic_key):
    sys.stdout.write(" Dimension: %d, Matrix number: %d     \r" % (D, widx + 1))
    sys.stdout.flush()

    W = ts[k]

    # Linear terms
    M[0, widx] = np.trace(W)  # \sum_{i} W_{ii}
    M[1, widx] = np.sum(W)  # \ sum_{i,j} W_{ij})

    # Quadratic terms

    W1 = W ** 2
    M[2, widx] = np.sum(W1)  # \sum_{i,j} W_{ij}^2
    W2 = W * W.T
    M[3, widx] = np.sum(W2)  # \sum_{i,j} W_{ij} W_{ji}

    Wd = np.diagonal(W)

    W3 = Wd * W.T
    M[4, widx] = np.sum(W3)  # \sum_{i,j} W_{ii} W_{ij}
    W4 = Wd * W
    M[5, widx] = np.sum(W4)  # \sum_{i,j} W_{ii} W_{ji}
    W5 = np.dot(W.T, W)
    M[6, widx] = np.sum(W5)  # \sum_{i,j,k} W_{ij} W_{ik}
    W6 = np.dot(W, W.T)
    M[7, widx] = np.sum(W6)  # \sum_{i,j,k} W_{ij} W_{kj}
    W7 = np.dot(W, W)
    M[8, widx] = np.sum(W7)  # \sum_{i,j,k} W_{ij} W_{jk}

    M[9, widx] = np.sum(W) ** 2  # \sum_{i,j,k,l} W_{ij} W_{kl}

    M[10, widx] = np.trace(W1)  # \sum_{i} W_{ii}^2

    M[11, widx] = np.trace(W) ** 2  # \sum_{i,j,k} W_{ii} W_{jj}

    M[12, widx] = (np.sum(W) * np.sum(Wd))  # \sum_{i,j,k} W_{ii} W_{jk}

    # Higher order matrix sums

    # Cubic terms

    W9 = W ** 3
    M[13, widx] = np.trace(W9)  # \sum_{i} W_{ii}^3  :  1-node case - Graph 1

    M[14, widx] = np.sum(W9)  # \sum_{i,j} W_{ij}^3   :  2-node case - Graph 2
    W10 = np.dot(np.dot(W, W), W)
    M[15, widx] = np.trace(W10)  # \sum_{i,j,k} W_{ij} W_{jk} W_{ki} : 3 node case - Graph 3

    M[16, widx] = np.sum((W.sum(axis=1)) * (W.sum(axis=0)) * Wd)  # \sum_{i,j,k} M_{ij}M_{jj}M_{jk} : 3 node-two case
    # - Graph 4

    M[17, widx] = np.sum(W) * (np.trace(W) ** 2)  # \sum_{i,j,k,l} W_{ij}W_{kk}W_{ll} : 4-node case - Graph 5

    M[18, widx] = (np.trace(W) * np.sum(W7))  # \sum_{i,j,k,l} M_{ij}M_{jk}M_{ll} : # 4 node-two case - Graph 6

    M[19, widx] = (np.sum(W) ** 2) * (np.trace(W))  # \sum_{i,j,k,l,m} W_{ij}W_{kl}W_{mm} : 5 node case - Graph 7

    M[20, widx] = np.sum(W) ** 3  # \sum_{i,j,k,l,m,n} W_{ij} W_{kl} W_{mn} :  6-node case - Graph 8

    # Quartic terms
    M[21, widx] = (np.sum(W)) ** 3 * np.trace(W)  # \sum_{i,j,k,l,m,n,o,o}} W_{ij} W_{kl} W_{mn} W_{oo} : 7 node case
    # - Graph 9

    M[22, widx] = np.sum(W) ** 4  # \Sum_{i,j,...,p} W_{ij} W_{kl} W_{mn} W_{op} 8 node case - Graph 10

ts.close()

# M1 holds the expectation values corresponding to each matrix sum
M1 = np.mean(M, axis=1)

# Setting these expectation values as a list
exp_list = []
for k in range(M1.shape[0]):
    exp_list.append(M1[k])

print(" \n Experimental expectation values:")
print(exp_list)

# Separating the linear and quadratic exp. vals. (mat_exp) from the cubic and quartic exp. vals. (mat_exp_cuqu)
mat_exp = exp_list[0:13]
mat_exp_cuqu = exp_list[13:24]

# Defining a function that contains all 13 equations in the PIGMM paper. By including the experimental expectation
# values, the 13 parameters of the model can be solved for.


def f(y):
    f1 = y[0] + (np.sqrt(D-1))*y[1] - float(mat_exp[0])  # <sum_{i} M_{ii}>
    f2 = D*y[0] - float(mat_exp[1])  # <sum_{i,j} M_{ij}>
    f3 = y[0]**2 + y[1]**2 + y[2] + y[4] + (D-1)*y[8] + (D-1)*y[10] + (D-1)*y[5] + ((D*(D-3))/2)*y[11] +\
        (((D-1)*(D-2))/2)*y[12] - float(mat_exp[2])  # <sum_{i,j} M_ij M_ij>
    f4 = ((D*(D-3))/2)*y[11] - (((D-1)*(D-2))/2)*y[12] + 2*(D-1)*y[6] + (D-1)*y[10] + y[2] + y[4] + y[0]**2 +\
        y[1]**2 - float(mat_exp[3])  # <sum_{i,j} M_ij M_ji>
    f5 = y[2] + np.sqrt(D-1)*y[3] + (D-1)*y[6] + (D-1)*y[8] + (D-1)*(np.sqrt(D-2))*y[9] + y[0]**2 +\
        y[0]*y[1]*(np.sqrt(D-1)) - float(mat_exp[4])  # <sum_{i,j} M_ii M_ij>
    f6 = y[2] + np.sqrt(D-1)*y[3] + (D-1)*y[6] + (D-1)*y[5] + (D-1)*(np.sqrt(D-2))*y[7] + y[0]**2 +\
        y[0]*y[1]*(np.sqrt(D-1)) - float(mat_exp[5])  # <sum_{i,j} M_ii M_ji>
    f7 = D*y[2] + D*(D-1)*y[8] + D*(y[0]**2) - float(mat_exp[6])  # <sum_{i,j,k} M_ij M_ik>
    f8 = D*y[2] + D*(D-1)*y[5] + D*(y[0]**2) - float(mat_exp[7])  # <sum_{i,j,k} M_ij M_kj>
    f9 = D*y[2] + D*(D-1)*y[6] + D*(y[0]**2) - float(mat_exp[8])  # <sum_{i,j,k} M_ij M_jk>
    f10 = (D**2)*y[2] + (D**2)*(y[0]**2) - float(mat_exp[9])  # <sum_{i,j,k,l} M_ij M_kl>
    f11 = (D**-1)*y[2] + ((D-1)/D)*y[4] + 2*((np.sqrt(D-1))/D)*y[3] + ((D-1)/D)*y[5] + ((D-1)/D)*y[8] +\
        ((D-1)/D)*(D-2)*y[10] + 2*((D-1)/D)*y[6] + 2*((D-1)/D)*(np.sqrt(D-2))*y[7] +\
        2*((D-1)/D)*(np.sqrt(D-2))*y[9] + ((y[0]**2)/D) + 2*((np.sqrt(D-1))/D)*y[0]*y[1] +\
        ((D-1)/D)*(y[1]**2) - float(mat_exp[10])  # <sum_{i} M_ii M_ii OR (M_ii)^2>
    f12 = y[2] + (D-1)*y[4] + 2*(np.sqrt(D-1))*y[3] + y[0]**2 + 2*(np.sqrt(D-1))*y[0]*y[1] +\
        (D-1)*(y[1]**2) - float(mat_exp[11])  # <sum_{i,j} M_ii M_jj>
    f13 = D*y[2] + D*(np.sqrt(D-1))*y[3] + D*(y[0]**2) +\
        D*(np.sqrt(D-1))*y[0]*y[1] - float(mat_exp[12])  # <sum_{i,j,k} M_{ii} M_{jk}>

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13]


x = fsolve(f, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

# x is an array containing the solved model parameters.
# Each parameter as defined in PIGMM is associated to each x[i] as follows:
# x[0] = tilde(mu)_1
# x[1] = tilde(mu)_2
# x[2] = (LamV0^-1)_11
# x[3] = (LamV0^-1)_12
# x[4] = (LamV0^-1)_22
# x[5] = (LamVH^-1)_11
# x[6] = (LamVH^-1)_12
# x[7] = (LamVH^-1)_13
# x[8] = (LamVH^-1)_22
# x[9] = (LamVH^-1)_23
# x[10] = (LamVH^-1)_33
# x[11] = (LamV2^-1)
# x[12] = (LamV3^-1)

# k describes how well the optimisation using fsolve has worked. If the process fails, the user is notified.
opt_check = np.isclose(f(x), [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
for p in opt_check:
    if not p:
        print("\n Parameter optimisation: Unsuccessful, fsolve procedure has failed.")
        sys.exit()
    else:
        continue
print("\n Parameter optimisation: Successful")

print(" \n Parameter results at dimension " + str(D) + ":")
print(x)

# Theoretical expectation values - the following graph labelling system is used to match that of the GTMDS paper.

# G1: Exp Val for sum_{i} M_{ii}^3
G1 = 3*((x[0]/D) + ((np.sqrt(D-1))/D)*x[1]) * ((x[2]/D) + ((D-1)/D)*x[4] + 2*(np.sqrt(D-1)/D)*x[3] + ((D-1)/D)*x[5] +\
    ((D-1)/D)*x[8] + ((D-1)/D)*(D-2)*x[10] + 2*((D-1)/D)*x[6] + 2*((D-1)/D)*(np.sqrt(D-2))*x[7] +\
    2*((D-1)/D)*(np.sqrt(D-2))*x[9]) + (D**-2)*((x[0] + (np.sqrt(D-1))*x[1])**3)

# G2: Exp Val for sum_{i,j} M_{ij}^3
G2 = (x[0]**3)/D + (3/D)*x[0]*(x[1]**2) + ((D-2)/(D*(np.sqrt(D-1))))*(x[1]**3) + 3*(x[0]/D)*(x[2] + x[4] +\
    (D-1)*x[8] + (D-1)*x[10] + (D-1)*x[5] + ((D*(D-3))/2)*x[11] + (((D-1)*(D-2))/2)*x[12]) +\
    3*x[1]*x[4]*((D-2)/(D*(np.sqrt(D-1)))) + (6*x[1]*x[3])/D + 3*x[1]*x[10]*((D-3)/D)*(np.sqrt(D-1)) +\
    6*x[1]*x[6]*((np.sqrt(D-1))/D) + 6*x[1]*x[7]*((np.sqrt((D-1)*(D-2)))/D) +\
    6*x[1]*x[9]*((np.sqrt((D-1)*(D-2)))/D) + 3*x[1]*x[11]*((-D**2 + 3*D)/(2*D*np.sqrt(D-1))) -\
    (3/2)*x[1]*x[12]*(((D-2)*np.sqrt(D-1))/D)

# G3: Exp Val for sum_{i,j,k} M_{ij} M_{jk} M_{ki}
G3 = x[0]**3 + (x[1]**3)/(np.sqrt((D-1))) + 3*x[0]*(x[2] + (D-1)*x[6]) + 3*(x[1]/(np.sqrt(D-1)))*x[4] +\
    3*x[1]*(np.sqrt(D-1))*x[10] + 3*(x[1])*(np.sqrt(D-1))*x[6] + 3*x[1]*x[11]*((D*(D-3))/(2*(np.sqrt(D-1)))) -\
    3*x[1]*x[12]*(((D-2)*(np.sqrt(D-1)))/2)

# G4: Exp Val for sum_{i,j,k} M_{ij} M_{jj} M_{jk}
G4 = x[0]*x[2] + np.sqrt(D-1)*x[0]*x[3] +(D-1)*x[0]*x[5] + (D-1)*x[0]*x[6] +(D-1)*np.sqrt(D-2)*x[0]*x[7] + x[0]*x[2] +\
     (D-1)*x[0]*x[6] + np.sqrt(D-1)*x[1]*x[2] + ((D-1)**(3/2))*x[1]*x[6] + x[0]*x[2] + np.sqrt(D-1)*x[0]*x[3] +\
     (D-1)*x[0]*x[6] + x[0]*(D-1)*x[8] + (D-1)*np.sqrt(D-2)*x[0]*x[9] + (x[0]**3) + (x[0]**2)*x[1]*np.sqrt(D-1)

# G5: Exp value for sum_{i,j,k,l} M_{ij}M_{kk}M_{ll}
G5 = 2*(D*x[2] + D*(np.sqrt(D-1))*x[3])*(x[0] + np.sqrt(D-1)*x[1]) + (x[2] + (D-1)*x[4] +\
     2*np.sqrt(D-1)*x[3] + ((x[0] + np.sqrt(D-1)*x[1])**2)) * D*x[0]

# G6: Exp value for sum_{i,j,k,l} M_{ij}M_{jk}M_{ll}
G6 = 3*D*x[0]*x[2] + D*np.sqrt(D-1)*x[2]*x[1] + D*(D-1)*x[6]*x[0] + D*((D-1)**(3/2))*x[6]*x[1] +\
     2*D*(np.sqrt(D-1))*x[0]*x[3] + D*(x[0]**3) + D*np.sqrt(D-1)*(x[0]**2)*x[1]

# G7: Exp val for sum_{i,j,k,l,m} M_{ij}M_{kl}M_{mm}
G7 = ((D**2)*x[2])*(x[0] + np.sqrt(D-1)*x[1]) + 2*(D*x[2] + D*np.sqrt(D-1)*x[3])*D*x[0] + ((D*x[0])**2)*(x[0] +\
     np.sqrt(D-1)*x[1])

# G8: Exp Val for sum_{i,j,k,l,m,n} M_{ij} M{kl} M{mn}
G8 = 3*x[0]*(D**3)*x[2] + (x[0]**3)*(D**3)

# G9: Exp Val for sum_{i_(1,2,3,4,5,6,7)} M_{i_(1,2)} M_{i_(3,4)} M_{i_(5,6)}M_{i_(7,7)}
G9 = 3*((D**2)*x[2]*(D*x[2] + D*np.sqrt(D-1)*x[3])) + 3*((D**2)*x[2]*(D*x[0])*(x[0] + np.sqrt(D-1)*x[1]) +\
    (D*x[2] + D*np.sqrt(D-1)*(x[3]))*(D*x[0])**2) + ((D*x[0])**3)*(x[0] + np.sqrt(D-1)*x[1])

# G10: Exp Val for sum_{i_(1,2,3,4,5,6,7,8)} M_{i_(1,2)} M_{i_(3,4)} M_{i_(5,6)}M_{i_(7,8)}
G10 = 3*(D**4)*(x[2]**2) + 6*(D**4)*x[2]*(x[0]**2) + (D*x[0])**4

# The ratios for each graph are then printed.
print("\n Theoretical vs. Experimental graph ratio results:")

print(" G1 ratio:", G1/float(mat_exp_cuqu[0]), "\n G2 ratio:", G2/float(mat_exp_cuqu[1]),
      "\n G3 ratio:", G3/float(mat_exp_cuqu[2]), "\n G4 ratio:", G4/float(mat_exp_cuqu[3]),
      "\n G5 ratio:", G5/float(mat_exp_cuqu[4]), "\n G6 ratio:", G6/float(mat_exp_cuqu[5]),
      "\n G7 ratio:", G7/float(mat_exp_cuqu[6]), "\n G8 ratio:", G8/float(mat_exp_cuqu[7]),
      "\n G9 ratio:", G9/float(mat_exp_cuqu[8]), "\n G10 ratio:", G10/float(mat_exp_cuqu[9]))

# Calculation of the convergence criteria.
print("\n Convergence Criteria:")


v0_power_of_neg_1 = np.array([[x[2], x[3]],
                              [x[3], x[4]]])

V0 = np.linalg.inv(v0_power_of_neg_1)
det_V0 = np.linalg.det(V0)
print("Criterion 1:", det_V0)

vh_power_of_neg_1 = np.array([[x[5], x[6], x[7]],
                              [x[6], x[8], x[9]],
                              [x[7], x[9], x[10]]])

VH = np.linalg.inv(vh_power_of_neg_1)
det_VH = np.linalg.det(VH)
print("Criterion 2:", det_VH)

Lam_V2 = (x[11]**-1)
Lam_V3 = (x[12]**-1)
print("Criterion 3:", Lam_V2)
print("Criterion 4:", Lam_V3)

if det_V0 >= 0 and det_VH >= 0 and Lam_V2 >= 0 and Lam_V3 >= 0:
    print("\n Convergence criteria test: Successful")
else:
    print("\n Convergence criteria test: Unsuccessful")
