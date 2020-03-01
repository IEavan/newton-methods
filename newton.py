import numpy as np


def pure_newton(func, grad, hessian, initial, tolerance, max_steps):
    iterations = 0
    x = initial

    while grad(x).dot(grad(x)) > tolerance ** 2 and iterations < max_steps:
        direction = np.linalg.solve(hessian(x), -grad(x))
        x += direction
        iterations += 1

        if x.dot(x) > 1e20:
            print("Failed to Converge")
            return iterations, None
    return iterations, x


def func_g(x):
    return np.sqrt(x[0] ** 2 + 1) + np.sqrt(x[1] ** 2 + 1)


def grad_g(x):
    return np.array([x[0] / np.sqrt(x[0] ** 2 + 1), x[1] / np.sqrt(x[1] ** 2 + 1)])


def hess_g(x):
    return np.array([[np.sqrt(x[0] ** 2 + 1) / (x[0] ** 4 + 2 * x[0] ** 2 + 1), 0],
                     [0, np.sqrt(x[1] ** 2 + 1) / (x[1] ** 4 + 2 * x[1] ** 2 + 1)]])


if __name__ == "__main__":
    print("Iterations: {} | Min: {}".format(
        *pure_newton(func_g, grad_g, hess_g, np.array([1.0,1.0]), 1e-8, 1e3)))
    print("Iterations: {} | Min: {}".format(
        *pure_newton(func_g, grad_g, hess_g, np.array([10.0,10.0]), 1e-8, 1e3)))
