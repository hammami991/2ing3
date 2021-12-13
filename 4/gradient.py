#----------------------------------------------------------------------------------------#
# Libraries
import sympy
from sympy import *

#----------------------------------------------------------------------------------------#
# Gradient Descent Implementation
class GradientDescent_RN():
    def __init__(self, l_rate=0.01, max_iter=1000, precision=1e-5, function=None,variables=None):
        self.l_rate = l_rate  # learning rate
        self.max_iter = max_iter  # Nb max d'iteration
        self.precision = precision  # stop prev_step_sizeition = precision
        self.function = function # function 
        self.variables = variables # symbols x,y,z ...

    # Input : Array of values : expl = (1,2,3,4)
    # Input : symbol_deriv = exemple calculate partial deriv in x 
    def _partial_derivative(self, symbol_deriv, values):
        if self.function and self.variables and values:
            dif1 = diff(self.function,symbol_deriv)
            new_list_tuple = []
            i = 0
            for ar in values:
                new_list_tuple.append((self.variables[i],ar))
                i+=1
            res = dif1.subs(new_list_tuple)
        del new_list_tuple
        return res

    def _calculate_fct(self, values):
        if self.function and self.variables and values:
            new_list_tuple = []
            i = 0
            for ar in values:
                new_list_tuple.append((self.variables[i],ar))
                i+=1
        return self.function.subs(new_list_tuple)

    def start_gradient(self):
        n_dimension = len(self.variables)
        listofvars = [1.1] * n_dimension
        res = self._calculate_fct(listofvars)  # calculate f(x1,x2,..) = res
        prev_step_size = 1  # start with prev_step_size greater than eps (assumption)
        nb_iter = 0  # init iteration counter
        prev_res = res  # init previous res
        while prev_step_size > self.precision and nb_iter < self.max_iter:
            for i in range(n_dimension):
                listofvars[i] = listofvars[i] - self.l_rate * self._partial_derivative(self.variables[i],listofvars)  # Make a small step down Xn
            res = self._calculate_fct(listofvars)
            nb_iter = nb_iter + 1  # iteration count
            prev_step_size = abs(prev_res - res)  # Change in res
            prev_res = res  # Store current res value in prev_res
        if nb_iter == self.max_iter:
            print("Maximum Iterations Exceeded : Could not find local Mininmun.")
            return
        print("Iterations : ", nb_iter)
        new_list = [str(self.variables[idx]) + " = "+ str(round(val,2)) for idx, val in enumerate(listofvars)]
        print("Found Local Minimum = {}; in points {}".format(res,new_list))


#----------------------------------------------------------------------------------------#
# Functions

# Function fct2
def fct2():
    x, y = symbols('x y')
    fct2_exp = x**2 + y**4
    return sympy.expand(fct2_exp), x, y

# Rosenbrock Function
def fct3():
    x, y = symbols('x y')
    fct3_exp = (1-x)**2 + 100 * (y-x**2)**2
    return sympy.expand(fct3_exp), x, y


#----------------------------------------------------------------------------------------#
# Main

if __name__ == "__main__":
    fn = fct2()
    gradient_fct2 = GradientDescent_RN(function=fn[0],variables=fn[1::])
    gradient_fct2.start_gradient()
