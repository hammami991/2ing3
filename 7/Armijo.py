import numpy as np
from scipy.stats import ortho_group
from matplotlib import pyplot as plt

def armijo(n, seed=135, max_iters=200):

    def f(x):
        """
        calculate f(x) with input x
        """
        return np.asscalar(1 / 2 * x.T * A * x + b.T * x)

    def df(x):

        """
        calculate f'(x) with input x
        """
        return A * x + b

    def grad_desc_const(alpha):
        """
        gradient descent with constant step size alpha
        """
        x = np.matrix(np.zeros((n, 1)))
        fun_values_const = [f(x)]
        for i in range(max_iters-1):
            # calculate derivative
            d = df(x)
            # update x
            x -= alpha * d
            # get new function value
            fun_values_const.append(f(x))

    def grad_desc_exact():
        """
        gradient descent with exact line minimization
        """
        x = np.matrix(np.zeros((n, 1)))
        fun_values_exact = [f(x)]
        for i in range(max_iters-1):
            # calculate derivative
            d = df(x)
            # calculate step size
            alpha = np.asscalar(d.T * d / (d.T * A * d))
            # update x
            x -= alpha * d
            # get new function value
            fun_values_exact.append(f(x))
            return fun_values_exact

    def grad_desc_armijo(alpha, beta=0.5, sigma=0.9):
        """
        gradient descent with Armijo step size rule
        """
        fun_values_exact = grad_desc_exact()
        x = np.matrix(np.zeros((n, 1)))
        fun_values_armijo = [f(x)]
        for i in range(max_iters-1):
            # calculate derivative
            d = df(x)
            # backtracking line search
            cur_alpha = alpha
            cur_value = f(x - cur_alpha * d)
            while cur_value <= fun_values_exact[-1] + sigma * cur_alpha * d.T * d:
                cur_alpha *= beta
                cur_value = f(x - cur_alpha * d)
            # update x
            x -= cur_alpha * d
            # get new function value
            fun_values_armijo.append(cur_value)
        return fun_values_armijo

    np.random.seed(seed)
    n = n
    b = np.matrix(2 * np.random.rand(n, 1) - 1)
    U = np.matrix(ortho_group.rvs(dim=n))
    D = np.matrix(np.diagflat(np.random.rand(n)))
    A = U.T * D * U
    max_iters = max_iters
    results = grad_desc_armijo(alpha=1)
    plt.plot(range(200), results, label='Armijo')
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Function Value")
    plt.show()
    return results


# create my random function
print(armijo(100))
# my_fun = quad_rand(100)
# run graident descent
# my_fun.grad_desc_const(alpha=0.3)
# my_fun.grad_desc_exact()
# my_fun.grad_desc_armijo(alpha=1)
# # plot
# plt.plot(range(200), my_fun.fun_values_const, label='Constant')
# plt.plot(range(200), my_fun.fun_values_exact, label='Exact Min')
# plt.plot(range(200), my_fun.fun_values_armijo, label='Armijo')
# plt.legend(loc="best")
# plt.xlabel("Iterations")
# plt.ylabel("Function Value")
# plt.show()