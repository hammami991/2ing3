from scipy.optimize import minimize,LinearConstraint,Bounds
import numpy as np
import numdifftools as nd

from scipy.optimize import linprog
import time

def min_fun(x):
    return (x[0]-1)**2 + (x[1]-4)**2


class frank_wolfe():

    def __init__(self, min_fun, A, b, bounds, x0, iterations=100):
        self.min_fun = min_fun
        self.A = A
        self.b = b
        self.bounds = bounds
        self.x0 = x0
        self.iterations = iterations
        self.x_min = []
        self.f_min = 0
        self.x_t = []
        self.s_t = []
        self.f_t = []
        self.violation = 0
        self.time = 0

    def __repr__(self):
        out = 'f_min: ' + str(self.f_min) + '\n' + \
              'x_min: ' + str(self.x_min) + '\n' + \
              'violation: ' + str(self.violation) + '\n' + \
              'time: ' + str(self.time)
        return out

    def optimize(self):
        x = self.x0
        t1 = time.time()
        for i in range(0, self.iterations):
            gamma = 2 / (i + 2)
            grad_def = nd.Gradient(self.min_fun)
            grad = grad_def(x)
            update = linprog(grad, A_ub=self.A, b_ub=self.b, A_eq=None, b_eq=None, bounds=self.bounds,
                             method='interior-point', callback=None, x0=None,
                             options={'sym_pos': False, 'lstsq': True})

            s = update.x
            self.s_t.append(s)
            x = x + gamma * (s - x)
            self.f_t.append(self.min_fun(x))
            self.x_t.append(x)
        t2 = time.time()
        self.time = t2 - t1

        constraints = np.dot(self.A, x) - self.b
        self.violation = np.sum([i for i in constraints if i > 0])
        self.x_min = x
        self.f_min = self.f_t[-1]
        return self

constr_num = 2
var_num = 3

A = np.random.randint(-10,10, (constr_num,var_num))
ub = 10*np.ones(constr_num)

bounds=[(0,10) for i in range(0,var_num)]
x0 = np.random.randint(0,10,(var_num))


iterations = 200
fw = frank_wolfe(min_fun,A,ub,bounds,x0,iterations)
results = fw.optimize()


bounds = Bounds(np.zeros(var_num), 10*np.ones(var_num))
lb = -np.inf*np.ones(constr_num)
linear_constraint = LinearConstraint(A, lb, ub)

res = minimize(min_fun, x0, method='trust-constr', jac=nd.Gradient(min_fun),
                constraints=linear_constraint, bounds=bounds,
                options={'verbose': 0,'gtol': 1e-8, 'disp': True})

f_star = [res.fun for i in range(0,len(results.f_t))]


    

