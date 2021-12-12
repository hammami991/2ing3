
import numpy
import numpy as np
import sympy
from sympy import symbols, Eq, solve, log
from matplotlib import pyplot as plt


def partial(element, function): #decalaration du fonction partiel ( element x,y ; funcion )
   
    partial_diff = function.diff(element) #dérive de f par rapprt x et y 

    return partial_diff # resultat 

def determat(partials_second, cross_derivatives, singular, symbols_list): #decalaration de det pour determiner min et max 
	
	det = partials_second[0].subs([(symbols_list[0], singular[symbols_list[0]]), (symbols_list[1], singular[symbols_list[1]])]) * partials_second[1].subs([(symbols_list[0], singular[symbols_list[0]]), (symbols_list[1], singular[symbols_list[1]])]) - (cross_derivatives.subs([(symbols_list[0], singular[symbols_list[0]]), (symbols_list[1], singular[symbols_list[1]])]))**2

	return det # resultat de determinant 

def gradient(partials): #declaration du la fonction gradient 
  
    grad = numpy.matrix([[partials[0]], [partials[1]]]) # partials fonction du bib numpy , partial : dérive de f  par rapport ( x et y ) // 0 et 1 ordre de x et y

    return grad #resultat du vecteur gradient


def gradient_to_zero(symbols_list, partials): #systeme d'equation = 0 // determiner point critique 
    
    partial_x = Eq(partials[0], 0) # partial par rapport x = 0
    partial_y = Eq(partials[1], 0) # partial par rapport y = 0

    singular = solve((partial_x, partial_y), (symbols_list[0], symbols_list[1])) # solve : resolution du system , symobols_list : les point critique en x et en y 

    return singular # resultat des point critiques 


def hessian(partials_second, cross_derivatives): # declaration du matrice hessien (dérivé second de f (x,y) par rapport (x/y) ; dérivé second de (f(x)/x) et (f(y)/y))
   
    hessianmat = numpy.matrix([[partials_second[0], cross_derivatives], [cross_derivatives, partials_second[1]]]) # forme du matrice 

    return hessianmat #resultat du matrice hessein 


def main():
   
    x, y = symbols('x y') #les elements 
    symbols_list = [x, y] 
    function = 2*x**4 + 5*y**3 # exemple de fonction 
    partials, partials_second = [],[] # dérive  du f par rapport x et y  // dérive seccond de f  par rapport x et y 

    for element in symbols_list:
        partial_diff = partial(element, function) #appel de system du fonction dérive partiel 
        partials.append(partial_diff) # append : ajouter des éléments à une liste 

    grad = gradient(partials) # appel de fonction au vecteur gradient
    singular = gradient_to_zero(symbols_list, partials) #appel de fonction au fonction singular : pour les point critiques 

    cross_derivatives = partial(symbols_list[0], partials[1]) #appel de fonction pour dérivé second 

    for i in range(0, len(symbols_list)): #nombre de point critique possible selon la fonction 
        partial_diff = partial(symbols_list[i], partials[i]) # i = nombre des point critiques // 
        partials_second.append(partial_diff) # ajouter des éléments de f à la liste 

    hessianmat = hessian(partials_second, cross_derivatives) #appel au fonction hessien 
    det = determat(partials_second, cross_derivatives, singular, symbols_list) # calculer derterminnats du matrices hessien
    
    print("Vecteur Gradient du fonction : {0} est :\n {1}".format(function,grad)) # affichage de fontion + Vecteur gradient
    print("Déterminant au point critique {0} est :\n {1}".format(singular, det)) # afficher du point critiques  
    print("Matrice hessienne qui organise toutes les dérivées partielles secondes de la fonction {0} est :\n {1}".format(
        function,hessianmat)) # affichage de fontion + marice hessien 
    
    
main()  

 


