import numpy as np


def linear_regression(x: list[float], y: list[float], params: list[float], iterations: int,
                      learning_rate: float) -> tuple[list[float], tuple[float, float]]:
    a, b = params
    n = len(x)
    loss_array = []

    k = 0
    while k < iterations:

        loss = (1 / n) * sum((y - (a + b * x)) ** 2)
        loss_array.append(loss)

        d_a = (-2 / n) * sum((y - (a + b * x)))
        d_b = (-2 / n) *  sum((x * (y - (a + b * x))))

        a = a - learning_rate * d_a
        b = b - learning_rate * d_b

        k += 1

    return loss_array, (a, b)



def parabolic_regression(x: list[float], y: list[float], params: list[float], iterations: int,
                         learning_rate: float) -> tuple[list[float], tuple[float, float, float]]:
    a, b, c = params
    n = len(x)
    loss_array = []

    k = 0
    while k < iterations:

        loss = (1 / n) * sum((y - (a + b * x + c * x ** 2)))
        loss_array.append(loss)

        d_a = (-2 / n) * sum((y - (a + b * x + c * x ** 2)))
        d_b = (-2 / n) * sum((x * (y - (a + b * x + c * x ** 2))))
        d_c = (-2 / n) * sum((x ** 2 * (y - (a + b * x + c * x ** 2))))

        a = a - learning_rate * d_a
        b = b - learning_rate * d_b
        c = c - learning_rate * d_c

        k += 1


    return loss_array, (a, b, c)




def sixth_deg_regression(x: list[float], y: list[float], params: list[float], iterations: int,
                         learning_rate: float) -> tuple[list[float], tuple[float, float, float, float, float, float]]:
    a, b, c, d, e, f, g = params
    n = len(x)
    loss_array = []

    k = 0
    while k < iterations:
        loss = (1 / n) * sum((y - (a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5 + g * x ** 6)) ** 2)
        loss_array.append(loss)

        d_a = (-2 / n) * sum((y - (a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5 + g * x ** 6)))
        d_b = (-2 / n) * sum((x * (y - (a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5 + g * x ** 6))))
        d_c = (-2 / n) * sum((x ** 2 * (y - (a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5 + g * x ** 6))))
        d_d = (-2 / n) * sum((x ** 3 * (y - (a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5 + g * x ** 6))))
        d_e = (-2 / n) * sum((x ** 4 * (y - (a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5 + g * x ** 6))))
        d_f = (-2 / n) * sum((x ** 5 * (y - (a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5 + g * x ** 6))))
        d_g = (-2 / n) * sum((x ** 6 * (y - (a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5 + g * x ** 6))))

        a = a - learning_rate * d_a
        b = b - learning_rate * d_b
        c = c - learning_rate * d_c
        d = d - learning_rate * d_d
        e = e - learning_rate * d_e
        f = f - learning_rate * d_f
        g = g - learning_rate * d_g


        k += 1

    return loss_array, (a, b, c, d, e, f, g)

