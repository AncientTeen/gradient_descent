import numpy as np
from icecream import ic


def linear_regression(x: list[float], y: list[float], params: list[float], iterations: int,
                      learning_rate: float) -> tuple[list[float], tuple[float, float]]:
    a, b = params
    n = len(x)
    loss_array = []

    k = 0
    while k < iterations:
        # ic(a, b)
        pred = a + b * x
        loss = (1 / n) * sum((y - pred) ** 2)
        loss_array.append(loss)

        d_a = (-2 / n) * sum((y - pred))
        d_b = (-2 / n) * sum((x * (y - pred)))

        a = a - learning_rate * d_a
        b = b - learning_rate * d_b

        k += 1
    ic(a, b)
    return loss_array, (a, b)


def parabolic_regression(x: list[float], y: list[float], params: list[float], iterations: int,
                         learning_rate: float) -> tuple[list[float], tuple[float, float, float]]:
    a, b, c = params
    n = len(x)
    loss_array = []

    k = 0
    while k < iterations:
        # ic(a, b, c)
        pred = a + b * x + c * x ** 2
        loss = (1 / n) * sum((y - pred) ** 2)
        loss_array.append(loss)

        d_a = (-2 / n) * sum((y - pred))
        d_b = (-2 / n) * sum((x * (y - pred)))
        d_c = (-2 / n) * sum((x ** 2 * (y - pred)))

        a = a - learning_rate * d_a
        b = b - learning_rate * d_b
        c = c - learning_rate * d_c

        k += 1

    ic(a, b, c)
    return loss_array, (a, b, c)


def sixth_deg_regression(x: list[float], y: list[float], params: list[float], iterations: int,
                         learning_rate: float) -> tuple[list[float], tuple[float, float, float, float, float, float]]:
    a, b, c, d, e, f, g = params
    n = len(x)
    loss_array = []

    k = 0
    while k < iterations:
        pred = a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5 + g * x ** 6
        loss = (1 / n) * sum((y - pred) ** 2)
        loss_array.append(loss)

        d_a = (-2 / n) * sum((y - pred))  # ∂/∂a
        d_b = (-2 / n) * sum((x * (y - pred)))  # ∂/∂b
        d_c = (-2 / n) * sum((x ** 2 * (y - pred)))  # ∂/∂c
        d_d = (-2 / n) * sum((x ** 3 * (y - pred)))  # ∂/∂d
        d_e = (-2 / n) * sum((x ** 4 * (y - pred)))  # ∂/∂e
        d_f = (-2 / n) * sum((x ** 5 * (y - pred)))  # ∂/∂f
        d_g = (-2 / n) * sum((x ** 6 * (y - pred)))  # ∂/∂g

        a = a - learning_rate * d_a
        b = b - learning_rate * d_b
        c = c - learning_rate * d_c
        d = d - learning_rate * d_d
        e = e - learning_rate * d_e
        f = f - learning_rate * d_f
        g = g - learning_rate * d_g

        k += 1

    return loss_array, (a, b, c, d, e, f, g)
