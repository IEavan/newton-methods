import numpy as np
import itertools


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


def damped_newton(func, grad, hessian, alpha, beta, initial, tolerance, max_steps):
    iterations = 0
    x = initial

    while grad(x).dot(grad(x)) > tolerance ** 2 and iterations < max_steps:
        direction = np.linalg.solve(hessian(x), -grad(x))
        t = 1
        while func(x) - func(x + t * direction) < - alpha * t * grad(x).dot(direction):
            t *= beta
        x += t * direction
        iterations += 1

        if x.dot(x) > 1e20:
            print("Failed to Converge")
            return iterations, None

    return iterations, x


def hybrid_newton(func, grad, hessian, alpha, beta, initial, tolerance, max_steps):
    iterations = 0
    x = initial

    while grad(x).dot(grad(x)) > tolerance ** 2 and iterations < max_steps:
        if is_pos_def(hessian(x)):
            direction = np.linalg.solve(hessian(x), -grad(x))
        else:
            print("Bruh!")
            direction = - grad(x)

        t = 1
        while func(x) - func(x + t * direction) < - alpha * t * grad(x).dot(direction):
            t *= beta
        x += t * direction
        iterations += 1

        if x.dot(x) > 1e20:
            print("Failed to Converge")
            return iterations, None

    return iterations, x


def is_pos_def(A):
    return np.all(np.linalg.eigvals(A) > 0)


# Function G
def func_g(x):
    return np.sqrt(x[0] ** 2 + 1) + np.sqrt(x[1] ** 2 + 1)
def grad_g(x):
    return np.array([x[0] / np.sqrt(x[0] ** 2 + 1), x[1] / np.sqrt(x[1] ** 2 + 1)])
def hess_g(x):
    return np.array([[np.sqrt(x[0] ** 2 + 1) / (x[0] ** 4 + 2 * x[0] ** 2 + 1), 0],
                     [0, np.sqrt(x[1] ** 2 + 1) / (x[1] ** 4 + 2 * x[1] ** 2 + 1)]])


# Function H
def func_h(x):
    def h1(x): return -13 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1]
    def h2(x): return -29 + x[0] + ((1 + x[1]) * x[1] - 14) * x[1]
    return h1(x) ** 2 + h2(x) ** 2
def grad_h(x):
    return np.array([4*x[0] + 12*x[1]**2 - 32*x[1] - 84,
                     24*x[0]*x[1] - 32*x[0] + 12*x[1]**5 - 40*x[1]**4 +
                     8*x[1]**3 - 240*x[1]**2 + 24*x[1] + 864])
def hess_h(x):
    return np.array([[4, 24*x[1] - 32],
                     [24*x[1] - 32, 24*x[0] + 60*x[1]**4 - 160*x[1]**3 + 24*x[1]**2 - 480*x[1] + 24]])


if __name__ == "__main__":
    g = (func_g, grad_g, hess_g)
    h = (func_h, grad_h, hess_h)

    print("Pure Newton Method")
    print("Iterations: {} | Min: {}".format(
        *pure_newton(*g, np.array([1.0, 1.0]), 1e-8, 1e3)))
    print("Iterations: {} | Min: {}".format(
        *pure_newton(*g, np.array([10.0, 10.0]), 1e-8, 1e3)))

    print("\nDamped Newton Method")
    print("Iterations: {} | Min: {}".format(
        *damped_newton(*g, 0.5, 0.5, np.array([1.0, 1.0]), 1e-8, 1e3)))
    print("Iterations: {} | Min: {}".format(
        *damped_newton(*g, 0.5, 0.5, np.array([10.0, 10.0]), 1e-8, 1e3)))

    print("\nHybrid Newton Method")
    methods = [damped_newton, hybrid_newton]
    start_points = [[-50, 7], [20, 7], [20, -18], [5, -10]]
    for method, x0 in itertools.product(methods, start_points):
        print("Method: {} | Start: {} | Iterations: {} | Min: {}".format(
            method.__name__, x0, *method(*h, 0.5, 0.5, np.array(x0).astype(np.float64), 1e-5, 1e3)))
