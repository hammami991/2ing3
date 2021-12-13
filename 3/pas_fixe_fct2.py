import numpy

#Definition de la fonction 2
def fct2(x):
    y = numpy.asarray(x)
    return numpy.sum(y[0]**2+y[1]**4)

#Calcul du gradient de la fonction 2
def fct2Gradient(x):
    y = numpy.asarray(x)
    grad = numpy.zeros_like(y)
    grad[0] = 2*y[0]
    grad[1] = 4*(y[1]**3)
    return grad

#Definition de la fonction pour le calcul du gradient a pas fixe.
def Gradient_Pas_Fixe(f,f_grad,gradient_error,point_error,x0,Tolerance,NB_ITR):
    dimension = numpy.max(numpy.shape(x0))
    XArray = numpy.zeros([dimension,NB_ITR])
    fArray = numpy.zeros(NB_ITR)
    point_error_array = numpy.zeros(NB_ITR)
    gradient_error_array = numpy.zeros(NB_ITR)
    x = numpy.asarray(x0)
    xx = x
    grad = f_grad(x)
    for i in range(NB_ITR):
        x = x - Tolerance*f_grad(x)
        grad_x = f_grad(x)
        ff = f(x)
        XArray[:,i] = x
        fArray[i] = ff
        point_error_array[i] = numpy.linalg.norm(x - xx)
        gradient_error_array[i] = numpy.linalg.norm(grad)
        if i % 100 == 0: #Affichage des rèsultas chaque 100 itération
            print(f"Iteration={i+1}, x={x}, f(x)={f(x)}")
        if (point_error_array[i]<point_error)|(gradient_error_array[i]<gradient_error):
            break
        xx = x
    #Affichage du résultat final
    print("---------------------------------------------------------")
    print("Final Results:\n")
    print(f"x={x}\nIteration={i+1}\nf(x)={f(x)}")
    
    return {'XArray':XArray[:,0:i],'fArray':fArray[0:i],'point_error_array':point_error_array[0:i],'point_error_array':point_error_array[0:i]}


x0 = numpy.array([1.1,2.1]) 
point_error = 10**-5
gradient_error = 10**-5


#Appel de la fonction de calcul du gradient a pas fixe.
Gradient_Pas_Fixe(fct2,fct2Gradient,gradient_error,point_error,x0,10**-3,10000)
