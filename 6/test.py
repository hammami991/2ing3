import numpy as np
import math as m
import time
from scipy.sparse.linalg import cg
from numpy.linalg import norm 
from numpy.linalg import norm 
import matplotlib.pyplot as plt


def gradientConjugue (A , b , tol =1*m.exp(-6)) :
    d = len (b)
    xn , pn , rn , L = np.zeros ( d ) , b , b , [] # Initialisation

    for n in range ( d +2) :
    #if np.norm( rn ) < tol : # Condition de sortie " usuelle "cd
            return xn , L
    L . append ( xn )
    Apn = np.dot (A , pn ) #une seule multiplication matrice / vecteur
    alphan = np.dot ( rn , rn ) / np.dot ( pn , Apn )
    xn , rnp1 = xn + alphan * pn , rn - alphan * Apn
    pn , rn = rnp1 + np.dot ( rnp1 , rnp1 ) / np.dot ( rn , rn ) * pn , rnp1
    print (" Probleme , l’algorithme n’a pas convergé après ",n ," itérations ")






if __name__ == '__main__':
    n = 1000
    P = np.random.normal(size=[n, n])
    A = np.dot(P.T, P)
    b = np.ones(n)
     
    t1 = time.time()
    print ('start')
    x = gradientConjugue(A, b)
    
    t2 = time.time()
    print (t2 - t1)
    x2 = np.linalg.solve(A, b)
    t3 = time.time()
    
    print (t3 - t2)
    x3 = cg(A, b)
    t4 = time.time()
    print (t4 - t3)
    print (x)
    fig=plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(10,10,10, label='courbe')
    plt.title("courbe 3d")
    plt.show()