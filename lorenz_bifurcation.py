import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams['figure.figsize'] = (8, 5)

sigma = 10
rho = 28
beta = 8/3
time_horizon = 1.0
x0 = 1.0
y0 = 0.2
z0 = 1.0

z_star = rho - 1

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

stable = z > z_star
changes = np.where(np.diff(stable.astype(int)) != 0)[0]

plt.plot(t, x, "--", label="Exact x", )
plt.plot(t, y, "--", label="Exact y")
plt.plot(t, z, "--", label="Exact z", color="black", lw=2)

# plt.fill_between(t, z, min(min(x), min(y)), where=(z < z_star), color="green", alpha=0.2) 
# plt.fill_between(t, z, min(min(x), min(y)), where=(z > z_star), color="red", alpha=0.2)

y_min, y_max = plt.ylim()[0], plt.ylim()[1]

plt.fill_between(t, y_min, y_max, where=z < z_star, color="red", alpha=0.12, interpolate=True)
plt.fill_between(t, y_min, y_max, where=z >= z_star, color="green", alpha=0.12, interpolate=True)

plt.axhline(z_star, label="z*", color="red")

plt.xlabel("t")
plt.ylabel("state")
plt.title(f"Lorenz Bifurcation Crossings (T = {time_horizon}; x0 = {x0}, y0 = {y0}, z0 = {z0})")

plt.legend()
plt.show()