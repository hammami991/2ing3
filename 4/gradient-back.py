#----------------------------------------------------------------------------------------#
# Libraries
import sympy
from sympy import *

#----------------------------------------------------------------------------------------#
# Gradient Descent Implementation
class GradientDescent_R2():
    def __init__(self, l_rate=0.01, max_iter=1000, precision=1e-5, function=None):
        self.l_rate = l_rate  # learning rate
        self.max_iter = max_iter  # Nb max d'iteration
        self.precision = precision  # stop prev_step_sizeition = precision
        self.function, self.x, self.y = function

    def _partial_derivative_x(self, x_val, y_val):
        dif1 = diff(self.function, self.x)
        res = dif1.subs([(self.x, x_val), (self.y, y_val)])
        return res

    def _partial_derivative_y(self, x_val, y_val):
        dif2 = diff(self.function, self.y)
        res = dif2.subs([(self.x, x_val), (self.y, y_val)])
        return res

    def _calculate_fct(self, x_val, y_val):
        return self.function.subs([(self.x, x_val), (self.y, y_val)])

    def start_gradient(self):
        x1_0 = 1.0  # x1 start point
        x2_0 = 1.5  # x2 start point
        res = self._calculate_fct(x1_0, x2_0)  # calculate f(x1,x2) = res
        prev_step_size = 1  # start with prev_step_size greater than eps (assumption)
        nb_iter = 0  # init iteration counter
        prev_res = res  # init previous res
        while prev_step_size > self.precision and nb_iter < self.max_iter:
            tmp_x1_0 = x1_0 - self.l_rate * \
                self._partial_derivative_x(
                    x1_0, x2_0)  # Make a small step down x1
            tmp_x2_0 = x2_0 - self.l_rate * \
                self._partial_derivative_y(
                    x1_0, x2_0)  # Make a small step down x2
            x1_0 = tmp_x1_0
            x2_0 = tmp_x2_0
            res = self._calculate_fct(x1_0, x2_0)
            nb_iter = nb_iter + 1  # iteration count
            prev_step_size = abs(prev_res - res)  # Change in res
            prev_res = res  # Store current res value in prev_res
        if nb_iter == self.max_iter:
            print("Maximum Iterations Exceeded : Could not find local Mininmun.")
            return
        print("Iterations : ", nb_iter)
        print("Found Local Minimum = {} ; in (X={},Y={}).".format(
            res, round(x1_0, 2), round(x2_0, 2)))


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
    gradient_fct2 = GradientDescent_R2(function=fct2())
    print("[!] Executing Gradient Descent on fct2 : ")
    gradient_fct2.start_gradient()

    gradient_fct3 = GradientDescent_R2(function=fct3())
    print("[!] Executing Gradient Descent on fct3 : ")
    gradient_fct3.start_gradient()
