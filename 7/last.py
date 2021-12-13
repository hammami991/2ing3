from time import time

import numpy as np
from matplotlib import pyplot as plt


def armijo(max_iters=200, A=None, b=None):
    b = b if b is not None else np.matrix([[1], [2], [3], [4], [5]])
    A = A if A is not None else np.matrix([[3, -1, 0, 0, 0],
                                           [-1, 12, -1, 0, 0],
                                           [0, -1, 24, -1, 0],
                                           [0, 0, -1, 48, -1],
                                           [0, 0, 0, -1, 96]])

    n = len(b)

    def f(x):
        """
        calculate f(x) with input x
        """
        return (1 / 2 * np.dot((np.dot(x.T, A)), x) - np.dot(b.T, x)).item()

    def df(x):
        """
        calculate f'(x) with input x
        """
        return A * x + b

    def grad_desc_exact():
        """
        gradient descent with exact line minimization
        """
        x = np.matrix(np.zeros((n, 1)))
        fun_values_exact = [f(x)]
        for i in range(max_iters - 1):
            # calculate derivative
            d = df(x)
            # calculate step size
            alpha = (d.T * d / (d.T * A * d)).item()
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
        curr_iter = 1
        for i in range(max_iters - 1):
            # calculate derivative
            d = df(x)
            # backtracking line search
            cur_alpha = alpha
            cur_value = f(x + cur_alpha * d)
            while cur_value > f(x) + sigma * cur_alpha * d.T * d:
                cur_alpha *= beta
                cur_value = f(x + cur_alpha * d)
            # update x
            x -= cur_alpha * d
            # get new function value
            fun_values_armijo.append(cur_value)
            if cur_value == fun_values_armijo[-1]:
                curr_iter = curr_iter
            else:
                curr_iter += 1
        print('nombre d\'iteration Armijo ', curr_iter)
        return fun_values_armijo

    results = grad_desc_armijo(alpha=1)
    plt.plot(range(10), results[:10], label='Armijo rn')
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Function Value")
    plt.show()
    return results


def armijo_r2(f=None, dfx1=None, dfx2=None, t=1, count=1, x0=None, alpha=0.3, beta=0.8):
    f = f if f is not None else lambda x: ((x[0] - 1) ** 2 + (x[1] - 4) ** 2)
    dfx1 = dfx1 if dfx1 is not None else lambda x: (2 * x[0])
    dfx2 = dfx2 if dfx2 is not None else lambda x: (2 * x[1])
    x0 = x0 if x0 is not None else np.array([2, 3])

    def backtrack(x0, dfx1, dfx2, t, alpha, beta, count):
        fun_value = []
        while (f(x0) - (f(x0 - t * np.array([dfx1(x0), dfx2(x0)])) + alpha * t * np.dot(np.array([dfx1(x0), dfx2(x0)]),
                                                                                        np.array([dfx1(x0),
                                                                                                  dfx2(x0)])))) < 0:
            t *= beta
            fun_value.append(t)
            count += 1
        return t, count, fun_value

    t, count, fun_value = backtrack(x0, dfx1, dfx2, t, alpha, beta, count)
    plt.plot(range(len(fun_value)), fun_value, label='Armijo r2')
    plt.legend(loc="best")
    plt.xlabel("Iterations")
    plt.ylabel("Function Value")
    plt.show()
    print("\nfinal step size :", t, " \nnombre d'iteration Armijo: ", count)
    return fun_value


c1 = time()
print("armijo r2 :", armijo_r2())
print('temps de execution Armijo r2 : ', time() - c1)

c = time()
print("armijo rn :", armijo())
print('temps de execution Armijo : ', time() - c)
