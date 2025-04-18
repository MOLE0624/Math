#!/usr/bin/env python3

# ================================================================================
# File       : 1_linear_regression.py
# Author     : MOLE0624 (GitHub: https://github.com/MOLE0624)
# Description: Demonstrates linear regression using the least squares method,
#              with and without data centering.
# Date       : 2025-04-03
# ================================================================================

from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from scipy import stats


def min_sq(x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate the slope and intercept using the least squares method.

    Args:
        x (jnp.ndarray): Input feature values.
        y (jnp.ndarray): Target values.

    Returns:
        list: [slope, intercept]
    """
    x_bar, y_bar = jnp.mean(x), jnp.mean(y)
    beta_1 = jnp.dot(x - x_bar, y - y_bar) / jnp.linalg.norm(x - x_bar) ** 2
    beta_0 = y_bar - beta_1 * x_bar
    return (beta_1, beta_0)


def gen_points(num_data: int = 100) -> Tuple[jnp.ndarray, jnp.ndarray]:
    N: int = num_data
    key = jax.random.PRNGKey(42)
    key_a, key_x, key_noise = jax.random.split(key, 3)

    a = jax.random.normal(key_a, shape=(N,)) + 2.0  # slope ~ N(2, 1)
    b = jnp.array([randn()])  # intercept (scalar)
    x = jax.random.normal(key_x, shape=(N,))
    y = a * x + b + jax.random.normal(key_noise, shape=(N,))  # y = ax + b + noise
    return (x, y)


def center_coords(x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return (x - jnp.mean(x), y - jnp.mean(y))


def sample():
    N = 100

    x, y = gen_points(N)
    a1, b1 = min_sq(x, y)  # regression on raw data
    print(f"a1 (centered slope): {a1}, b1 (centered bias): {b1}")

    # Center the data
    xx, yy = center_coords(x, y)
    a2, b2 = min_sq(xx, yy)  # regression on centered data
    print(f"a2 (centered slope): {a2}, b2 (centered bias): {b2}")

    # Visualization (convert to NumPy)
    x_np = np.array(x)
    y_np = np.array(y)
    x_seq = np.arange(-5, 5, 0.1)
    y_pred = x_seq * float(a1) + float(b1)
    yy_pred = x_seq * float(a2) + float(b2)

    plt.scatter(x_np, y_np, c="black", label="Sampled Data")
    plt.axhline(y=0, c="black", linewidth=0.5)
    plt.axvline(x=0, c="black", linewidth=0.5)
    plt.plot(x_seq, y_pred, c="orange", label="Before Centering")
    plt.plot(x_seq, yy_pred, c="blue", label="After Centering")
    plt.legend(loc="upper left")
    plt.title("Linear Regression: Centered vs. Non-Centered")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


def sample2():
    N = 100
    # p = 2

    beta = jnp.array([1, 2, 3])
    x = randn(N, 2)
    y = beta[0] + beta[1] * x[:, 0] + beta[2] * x[:, 1] + randn(N)
    X = jnp.insert(x, 0, 1, axis=1)
    print(jnp.linalg.inv(X.T @ X) @ X.T @ y)


def sample3():
    x = jnp.arange(0, 20, 0.1)
    for i in range(1, 11):
        plt.plot(x, stats.chi2.pdf(x, i), label="{}".format(i))
    plt.legend(loc="upper right")
    plt.show()


# t-distribution
def sample4():
    x = jnp.arange(-10, 10, 0.1)
    plt.plot(
        x, stats.norm.pdf(x, 0, 1), label="Regular Distribution", c="black", linewidth=1
    )
    for i in range(1, 11):
        plt.plot(x, stats.t.pdf(x, i), label=f"df={i}", linewidth=0.8)
    plt.legend(loc="upper right")
    plt.title("How the t-distribution changes with degrees of freedom")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.grid(True)
    plt.show()


def sample5():
    N = 100
    p = 1
    iter_num = 100
    for i in range(iter_num):
        x = randn(N) + 2
        e = randn(N)
        y = x + 1 + e
        b_1, b_0 = min_sq(x, y)
        plt.scatter(b_0, b_1)
    plt.axhline(y=1.0, c="black", linewidth=0.5)
    plt.axvline(x=1.0, c="black", linewidth=0.5)
    plt.xlabel("beta_0")
    plt.ylabel("beta_1")
    plt.show()


def sample24():
    N = 100
    x = randn(N)
    y = randn(N)
    beta_1, beta_0 = min_sq(x, y)
    RSS = np.linalg.norm(y - beta_0 - beta_1 * x) ** 2
    RSE = np.sqrt(RSS / (N - 1 - 1))
    B_0 = (x.T @ x / N) / np.linalg.norm(x - np.mean(x)) ** 2
    B_1 = 1 / np.linalg.norm(x - np.mean(x)) ** 2
    se_0 = RSE * np.sqrt(B_0)
    se_1 = RSE * np.sqrt(B_1)
    t_0 = beta_0 / se_0
    t_1 = beta_1 / se_1
    p_0 = 2 * (1 - stats.t.cdf(np.abs(t_0), N - 2))
    p_1 = 2 * (1 - stats.t.cdf(np.abs(t_1), N - 2))

    print("intercept      :", beta_0, se_0, t_0, p_0)
    print("regression coef:", beta_1, se_1, t_1, p_1)

    from sklearn import linear_model

    reg = linear_model.LinearRegression()
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    reg.fit(x, y)

    print("regression coef beta_1:", reg.coef_, ", intercept beta_0:", reg.intercept_)

    import statsmodels.api as sm

    X = np.insert(x, 0, 1, axis=1)
    model = sm.OLS(y, x)
    res = model.fit()
    print(res.summary())


def sample25():
    N = 100
    r = 1000
    T = []
    for i in range(r):
        x = randn(N)
        y = randn(N)
        beta_1, beta_0 = min_sq(x, y)
        pre_y = beta_0 + beta_1 * x
        RSS = np.linalg.norm(y - beta_0 - beta_1 * x) ** 2
        RSE = np.sqrt(RSS / (N - 1 - 1))
        B_0 = (x.T @ x / N) / np.linalg.norm(x - np.mean(x)) ** 2
        B_1 = 1 / np.linalg.norm(x - np.mean(x)) ** 2
        se_1 = RSE * np.sqrt(B_1)
        T.append(beta_1 / se_1)
    plt.hist(T, bins=20, range=(-3, 3), density=True)
    x = np.linspace(-4, 4, 400)
    plt.plot(x, stats.t.pdf(x, 1))
    plt.title("Cases in which the null hypothesis is true")
    plt.xlabel("t value")
    plt.ylabel("Probability Density")
    plt.show()

    T = []
    for i in range(r):
        x = randn(N)
        y = 0.1 * x + randn(N)
        beta_1, beta_0 = min_sq(x, y)
        pre_y = beta_0 + beta_1 * x
        RSS = np.linalg.norm(y - beta_0 - beta_1 * x) ** 2
        RSE = np.sqrt(RSS / (N - 1 - 1))
        B_0 = (x.T @ x / N) / np.linalg.norm(x - np.mean(x)) ** 2
        B_1 = 1 / np.linalg.norm(x - np.mean(x)) ** 2
        se_1 = RSE * np.sqrt(B_1)
        T.append(beta_1 / se_1)
    plt.hist(T, bins=20, range=(-3, 3), density=True)
    x = np.linspace(-4, 4, 400)
    plt.plot(x, stats.t.pdf(x, 1))
    plt.title("Cases in which the null hypothesis is NOT true")
    plt.xlabel("t value")
    plt.ylabel("Probability Density")
    plt.show()


if __name__ == "__main__":
    # sample()
    # sample2()
    # sample3()
    # sample4()
    # sample5()
    # sample24()
    # sample25()
