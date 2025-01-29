import numpy as np
from icecream import ic

def linear_regression(x: list[float], y: list[float], params: list[float], iterations: int,
                              learning_rate: float, epsilon: float = 1e-8) -> tuple[list[float], tuple[float, float]]:
    """
    Finds linear regression parameters using the AdaGrad optimizer.

    Args:
        x (list[float]): Independent variable data points.
        y (list[float]): Dependent variable data points.
        params (list[float]): Initial parameters [a, b].
        iterations (int): Number of iterations for optimization.
        learning_rate (float): Initial learning rate for the optimizer.
        epsilon (float): Small constant for numerical stability.

    Returns:
        tuple: (loss_array, final_params)
            - loss_array (list[float]): List of loss values per iteration.
            - final_params (tuple[float, float]): Optimized parameters (a, b).
    """
    # Convert lists to NumPy arrays for vectorized operations
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    params = np.array(params, dtype=np.float64)

    a, b = params
    n = len(x)
    loss_array = []

    # Initialize accumulated squared gradients
    G = np.zeros_like(params)

    for k in range(1, iterations + 1):
        # Prediction
        pred = a + b * x

        # Compute loss (Root Mean Squared Error)
        loss = np.sqrt(np.mean((y - pred) ** 2))
        loss_array.append(loss)

        # Compute gradients
        error = y - pred
        d_a = (-2 / n) * np.sum(error)
        d_b = (-2 / n) * np.sum(x * error)
        grads = np.array([d_a, d_b])

        # Accumulate squared gradients
        G += grads ** 2

        # Update parameters
        adjusted_lr = learning_rate / (np.sqrt(G) + epsilon)
        params = params - adjusted_lr * grads

        # Assign updated parameters
        a, b = params

    ic(a, b)
    return loss_array, (a, b)


def parabolic_regression(x: list[float], y: list[float], params: list[float], iterations: int,
                                 learning_rate: float, epsilon: float = 1e-8) -> tuple[list[float], tuple[float, float, float]]:
    """
    Finds parabolic regression parameters using the AdaGrad optimizer.

    Args:
        x (list[float]): Independent variable data points.
        y (list[float]): Dependent variable data points.
        params (list[float]): Initial parameters [a, b, c].
        iterations (int): Number of iterations for optimization.
        learning_rate (float): Initial learning rate for the optimizer.
        epsilon (float): Small constant for numerical stability.

    Returns:
        tuple: (loss_array, final_params)
            - loss_array (list[float]): List of loss values per iteration.
            - final_params (tuple[float, float, float]): Optimized parameters (a, b, c).
    """
    # Convert lists to NumPy arrays for vectorized operations
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    params = np.array(params, dtype=np.float64)

    a, b, c = params
    n = len(x)
    loss_array = []

    # Initialize accumulated squared gradients
    G = np.zeros_like(params)

    for k in range(1, iterations + 1):
        # Prediction
        pred = a + b * x + c * (x ** 2)

        # Compute loss (Root Mean Squared Error)
        loss = np.sqrt(np.mean((y - pred) ** 2))
        loss_array.append(loss)

        # Compute gradients
        error = y - pred
        d_a = (-2 / n) * np.sum(error)
        d_b = (-2 / n) * np.sum(x * error)
        d_c = (-2 / n) * np.sum((x ** 2) * error)
        grads = np.array([d_a, d_b, d_c])

        # Accumulate squared gradients
        G += grads ** 2

        # Update parameters
        adjusted_lr = learning_rate / (np.sqrt(G) + epsilon)
        params = params - adjusted_lr * grads

        # Assign updated parameters
        a, b, c = params

    ic(a, b, c)
    return loss_array, (a, b, c)


def sixth_deg_regression(x: list[float], y: list[float], params: list[float], iterations: int,
                                 learning_rate: float, epsilon: float = 1e-8) -> tuple[list[float], tuple[float, float, float, float, float, float, float]]:
    """
    Finds sixth-degree regression parameters using the AdaGrad optimizer.

    Args:
        x (list[float]): Independent variable data points.
        y (list[float]): Dependent variable data points.
        params (list[float]): Initial parameters [a, b, c, d, e, f, g].
        iterations (int): Number of iterations for optimization.
        learning_rate (float): Initial learning rate for the optimizer.
        epsilon (float): Small constant for numerical stability.

    Returns:
        tuple: (loss_array, final_params)
            - loss_array (list[float]): List of loss values per iteration.
            - final_params (tuple[float, float, float, float, float, float, float]):
              Optimized parameters (a, b, c, d, e, f, g).
    """
    # Convert lists to NumPy arrays for vectorized operations
    # print("regression камшот для саши")  # Removed unnecessary print statement

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    params = np.array(params, dtype=np.float64)

    a, b, c, d, e, f, g = params
    n = len(x)
    loss_array = []

    # Initialize accumulated squared gradients
    G = np.zeros_like(params)
    ic(learning_rate, iterations)

    for k in range(1, iterations + 1):
        # Prediction
        pred = (a + b * x + c * (x ** 2) + d * (x ** 3) +
                e * (x ** 4) + f * (x ** 5) + g * (x ** 6))

        # Compute loss (Root Mean Squared Error)
        loss = np.sqrt(np.mean((y - pred) ** 2))
        loss_array.append(loss)

        # Compute gradients
        error = y - pred
        d_a = (-2 / n) * np.sum(error)
        d_b = (-2 / n) * np.sum(x * error)
        d_c = (-2 / n) * np.sum((x ** 2) * error)
        d_d = (-2 / n) * np.sum((x ** 3) * error)
        d_e = (-2 / n) * np.sum((x ** 4) * error)
        d_f = (-2 / n) * np.sum((x ** 5) * error)
        d_g = (-2 / n) * np.sum((x ** 6) * error)
        grads = np.array([d_a, d_b, d_c, d_d, d_e, d_f, d_g])

        # Accumulate squared gradients
        G += grads ** 2

        # Update parameters
        adjusted_lr = learning_rate / (np.sqrt(G) + epsilon)
        params = params - adjusted_lr * grads

        # Assign updated parameters
        a, b, c, d, e, f, g = params

    ic(a, b, c, d, e, f, g)
    return loss_array, (a, b, c, d, e, f, g)
