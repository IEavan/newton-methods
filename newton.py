import numpy as np

def pure_newton(func, grad, hessian, initial, tolerance, max_steps):
    iterations = 0
    x = initial

    while grad(x).dot(grad(x)) > tolerance ** 2 and iterations < max_steps:
        direction = np.linalg.solve(hessian(x), -grad(x))
        x += direction

    return iterations, x
