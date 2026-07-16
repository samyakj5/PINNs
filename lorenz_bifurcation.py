import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sigma = 10
rho = 28
beta = 8/3
time_horizon = 1.0
x0 = 1.0
y0 = 0.2
z0 = 1.0

z_star_bar = rho - 1

def lorenz(t, X, sigma, beta, rho):
    x, y, z = X

    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

solution = solve_ivp(
    lorenz, (0, time_horizon), (x0, y0, z0), args=(sigma, beta, rho), dense_output=True
)

t = np.linspace(0, time_horizon, 10000)
x, y, z = solution.sol(t)

plt.plot(t, x, "--", label="Exact x", )
plt.plot(t, y, "--", label="Exact y")
plt.plot(t, z, "--", label="Exact z", color="green")
plt.fill_between(t, z, min(min(x), min(y)), where=(z < z_star_bar), color="green", alpha=0.2) 
plt.fill_between(t, z, min(min(x), min(y)), where=(z > z_star_bar), color="red", alpha=0.2) 

plt.axhline(z_star_bar, label="Crossing", color="red")

plt.xlabel("t")
plt.ylabel("state")

plt.legend()
plt.show()