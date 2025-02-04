# import numpy as np
#
#
# def linear_regression(x: list[float], y: list[float], params: list[float], iterations: int, learning_rate: float) -> tuple[list[float], list[float]]:
#     """
#     Finds linear regression parameters using the AdaGrad optimizer.
#     """
#     epsilon = 1e-8
#
#     x = np.array(x, dtype=np.float64)
#     y = np.array(y, dtype=np.float64)
#     params = np.array(params, dtype=np.float64)
#
#     a, b = params
#     n = len(x)
#     loss_array = []
#
#     G = np.zeros_like(params)
#
#     for k in range(1, iterations + 1):
#         pred = a + b * x
#
#         loss = np.sqrt(np.mean((y - pred) ** 2))
#         loss_array.append(loss)
#
#         error = y - pred
#         d_a = (-2 / n) * np.sum(error)
#         d_b = (-2 / n) * np.sum(x * error)
#         grads = np.array([d_a, d_b])
#
#         G += grads ** 2
#
#         adjusted_lr = learning_rate / (np.sqrt(G) + epsilon)
#         params = params - adjusted_lr * grads
#
#         a, b = params
#     final_params = [a, b]
#
#     return loss_array, final_params
#
#
# def parabolic_regression(x: list[float], y: list[float], params: list[float], iterations: int, learning_rate: float, epsilon: float = 1e-8) -> tuple[list[float], list[float]]:
#     """
#     Finds parabolic regression parameters using the AdaGrad optimizer.
#     """
#     epsilon = 1e-8
#
#     x = np.array(x, dtype=np.float64)
#     y = np.array(y, dtype=np.float64)
#     params = np.array(params, dtype=np.float64)
#
#     a, b, c = params
#     n = len(x)
#     loss_array = []
#
#     G = np.zeros_like(params)
#
#     for k in range(1, iterations + 1):
#         pred = a + b * x + c * (x ** 2)
#
#         loss = np.sqrt(np.mean((y - pred) ** 2))
#         loss_array.append(loss)
#
#         error = y - pred
#         d_a = (-2 / n) * np.sum(error)
#         d_b = (-2 / n) * np.sum(x * error)
#         d_c = (-2 / n) * np.sum((x ** 2) * error)
#         grads = np.array([d_a, d_b, d_c])
#
#         G += grads ** 2
#
#         adjusted_lr = learning_rate / (np.sqrt(G) + epsilon)
#         params = params - adjusted_lr * grads
#
#         a, b, c = params
#
#     final_params = [a, b, c]
#     return loss_array, final_params
#
#
# def sixth_deg_regression(x: list[float], y: list[float], params: list[float], iterations: int, learning_rate: float, ) -> tuple[list[float], list[float]]:
#     """
#     Finds sixth-degree regression parameters using the AdaGrad optimizer.
#     """
#     epsilon = 1e-8
#
#
#     x = np.array(x, dtype=np.float64)
#     y = np.array(y, dtype=np.float64)
#     params = np.array(params, dtype=np.float64)
#
#     a, b, c, d, e, f, g = params
#     n = len(x)
#     loss_array = []
#
#     G = np.zeros_like(params)
#
#     for k in range(1, iterations + 1):
#         pred = (a + b * x + c * (x ** 2) + d * (x ** 3) +
#                 e * (x ** 4) + f * (x ** 5) + g * (x ** 6))
#
#         loss = np.sqrt(np.mean((y - pred) ** 2))
#         loss_array.append(loss)
#
#         error = y - pred
#         d_a = (-2 / n) * np.sum(error)
#         d_b = (-2 / n) * np.sum(x * error)
#         d_c = (-2 / n) * np.sum((x ** 2) * error)
#         d_d = (-2 / n) * np.sum((x ** 3) * error)
#         d_e = (-2 / n) * np.sum((x ** 4) * error)
#         d_f = (-2 / n) * np.sum((x ** 5) * error)
#         d_g = (-2 / n) * np.sum((x ** 6) * error)
#         grads = np.array([d_a, d_b, d_c, d_d, d_e, d_f, d_g])
#
#         G += grads ** 2
#
#         adjusted_lr = learning_rate / (np.sqrt(G) + epsilon)
#         params = params - adjusted_lr * grads
#
#         a, b, c, d, e, f, g = params
#     final_params = [a, b, c, d, e, f, g]
#
#     return loss_array, final_params

import numpy as np


def linear_regression(x: list[float], y: list[float], params: list[float], iterations: int, learning_rate: float) -> tuple[list[float], list[float]]:
    """
    Finds linear regression parameters using the AdaGrad optimizer.
    """
    epsilon = 1e-8

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    params = np.array(params, dtype=np.float64)

    a, b = params
    n = len(x)
    loss_array = []

    learning_rate_tau = 0.01 * learning_rate

    for k in range(1, iterations + 1):
        pred = a + b * x

        loss = np.sqrt(np.mean((y - pred) ** 2))
        loss_array.append(loss)

        error = y - pred
        d_a = (-2 / n) * np.sum(error)
        d_b = (-2 / n) * np.sum(x * error)
        grads = np.array([d_a, d_b])

        learning_rate_k = (1 - k / iterations) * learning_rate + k / iterations * learning_rate_tau

        params = params - learning_rate_k * grads

        a, b = params
    final_params = [a, b]

    return loss_array, final_params


def parabolic_regression(x: list[float], y: list[float], params: list[float], iterations: int, learning_rate: float, epsilon: float = 1e-8) -> tuple[list[float], list[float]]:
    """
    Finds parabolic regression parameters using the AdaGrad optimizer.
    """
    epsilon = 1e-8

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    params = np.array(params, dtype=np.float64)

    a, b, c = params
    n = len(x)
    loss_array = []

    learning_rate_tau = 0.01 * learning_rate

    for k in range(1, iterations + 1):
        # pred = a + b * x + c * (x ** 2)
        pred = a + b * (x - 5) + c * ((x - 5) ** 2)
        # pred = a + b * (x + 5) + c * ((x + 5) ** 2)

        loss = np.sqrt(np.mean((y - pred) ** 2))
        loss_array.append(loss)

        error = y - pred
        d_a = (-2 / n) * np.sum(error)
        d_b = (-2 / n) * np.sum(x * error)
        d_c = (-2 / n) * np.sum((x ** 2) * error)
        grads = np.array([d_a, d_b, d_c])

        learning_rate_k = (1 - k / iterations) * learning_rate + k / iterations * learning_rate_tau

        params = params - learning_rate_k * grads

        a, b, c = params

    final_params = [a, b, c]
    return loss_array, final_params


def sixth_deg_regression(x: list[float], y: list[float], params: list[float], iterations: int, learning_rate: float, ) -> tuple[list[float], list[float]]:
    """
    Finds sixth-degree regression parameters using the AdaGrad optimizer.
    """
    epsilon = 1e-8


    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    params = np.array(params, dtype=np.float64)

    a, b, c, d, e, f, g = params
    n = len(x)
    loss_array = []

    learning_rate_tau = 0.01 * learning_rate

    for k in range(1, iterations + 1):
        pred = (a + b * x + c * (x ** 2) + d * (x ** 3) +
                e * (x ** 4) + f * (x ** 5) + g * (x ** 6))

        loss = np.sqrt(np.mean((y - pred) ** 2))
        loss_array.append(loss)

        error = y - pred
        d_a = (-2 / n) * np.sum(error)
        d_b = (-2 / n) * np.sum(x * error)
        d_c = (-2 / n) * np.sum((x ** 2) * error)
        d_d = (-2 / n) * np.sum((x ** 3) * error)
        d_e = (-2 / n) * np.sum((x ** 4) * error)
        d_f = (-2 / n) * np.sum((x ** 5) * error)
        d_g = (-2 / n) * np.sum((x ** 6) * error)
        grads = np.array([d_a, d_b, d_c, d_d, d_e, d_f, d_g])

        learning_rate_k = (1 - k / iterations) * learning_rate + k / iterations * learning_rate_tau

        params = params - learning_rate_k * grads

        a, b, c, d, e, f, g = params
    final_params = [a, b, c, d, e, f, g]

    return loss_array, final_params