import string
from sympy import *
import sys
def initForGrad(pexp, ppoint):
    """
    Cette fonction permet d'initialiser des variables necessaires au differente fonction de gradient
    le paramètre pexp est l'expression de la fonction
    le paramètre ppoint: premier point X0
    le retour: liste des variable de pexp, nombre de variable, symbole p du pas, initialisation d'un gradient correspondant à pexp, [[(var1, val1), ...,(varn, valn)]
    """
    variables = pexp.free_symbols
    return variables, len(variables), Symbol('p'), [pexp.diff(var) for var in variables], list(zip(variables, ppoint))
def Xk(pvec, ppas, pgrad, pdim, pmod=0, pcond=1):
    """
    cette fonction renvoie le point X k+1 pour une regression
    le paramètre pvec: [(var1, val1), ...,(varn, valn)] tableau qui associe au valeur de ppnt le nom de la variable à laquel
   le paramètre ppas: indique le pas necessaire pour l'iteration suivante
    le paramètre pgrad: [derivepartielEnVar1, .. , derivepartielEnVarn] gradient de la fonction d'origine
   le paramètre pdim: n dimension du vecteur de la fonction
    le retour: [val1, .. , valn] point à l'iteration k+1 soit Xk+1
    """
    if pmod == 0:
        res = [pvec[i][1] - ppas * pgrad[i].subs(pvec) for i in range(pdim)]
    elif pmod == 2:
        res = [pvec[i][1] + ppas * pgrad[i] for i in range(pdim)]
    else:
        res = [(pvec[i][1] - ppas * (pgrad[i].subs(pvec) / pcond)).evalf() for i in range(pdim)]
    return res
def expPas(ppnt, pgrad, pvec, pdim):
    """
    cette fonction renvoie l'expression de phi en fonction de p soit un vecteur de même dimension que pvec. Cette expression
    sert par la suite à calculer le pas optimal
    le paramètre ppnt: [val1, .. , valn] point à partir duquel le pas optimal
    le paramètre pgrad: [derivepartielEnVar1, .. , derivepartielEnVarn] gradient de la fonction d'origine
    le paramètre pvec: [(var1, val1), ...,(varn, valn)] tableau qui associe au valeur de ppnt le nom de la variable à laquel
    elle est associé
    le paramètre pdim: n dimension du vecteur de la fonction
    le retour: expression de phi de p
    """
    return [parse_expr(str(ppnt[i]) + " - p * " + str(pgrad[i].subs(pvec))) for i in range(pdim)]
def pasOpti(pexprpas, pvec, pp):
    """
    cette fonction permettant de calculer le pas optimal pour Xk+1
    le paramètre pexprpas: expression de f
    le paramètre pvec: vecteur trouver pour phi de p
    le paramètre pp: symbole du pas
    le retour: est le pas optimisé sinon -1 si fonction a échoué
    """
    pas = solve(pexprpas.subs(pvec), pp)  # fin calcul du pas opti
    res = -1
    for e in pas:
        if e > 0:
            res = e
            break
    return res
def gradPOpti(p_exp, ppt, tolerance):
    """
    cette fonction effectue une descente de gradient a pas optimisé
    le paramètre p_exp: expression de la fonction sur laquel effectuer une descente de gradient
    le paramètre ppt: [val1, .. , valn] point de depart de la descente de gradient
    le paramètre tolerance: de type float est le seuil à partir duquel on decidera que l'aproximation est suffisante (par rapport a la norme
    du point trouvé
    le paramètre pverbose: int par defaut à 0 et si different de 0 alors le mode verbose est activé
    le retour: [val1, .. , valn] point au plus proche du minimum local
    """
    variables, size, p, grad, vec = initForGrad(p_exp, ppt)
    expas = expPas(ppt, grad, vec, size)
    pas = pasOpti(p_exp, list(zip(variables, expas)), p)
    XK1 = Xk(vec, pas, grad, size)
    vec = list(zip(variables, XK1))
    cond = Matrix([grad[i].subs(vec) for i in range(size)]).norm()
    nombre_itter=1;
    while cond > tolerance:
        nombre_itter=nombre_itter+1
        print("X{} : {}".format( nombre_itter, XK1))
        expas = expPas(ppt, grad, vec, size)
        pas = pasOpti(p_exp, list(zip(variables, expas)), p)
        if pas == -1:
            break
        XK1 = Xk(vec, pas, grad, size)
        vec = list(zip(variables, XK1))
        cond = Matrix([grad[i].subs(vec) for i in range(size)]).norm()
    print ("Pas Optimal",pas)
    print("nombre d'ittération",nombre_itter)
    return XK1
"exemple de cours 'algo' avec resultat juste"
f = parse_expr("x**2+y**2+z**2-3*y*z-6*x*y")
b=0.0001
x0=[1,1,1]
res = gradPOpti(f, x0, float(b))
variables = f.free_symbols
vec = list(zip(variables, res))
print("point au plus proche du minimum local {}".format(f.subs(vec)))
print("approximation trouvé : {}".format(vec))
