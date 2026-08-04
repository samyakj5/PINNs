import os
import sys
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = (sys.platform == "darwin")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
from scipy.integrate import solve_ivp

"""Latin hypercube sampling of collocation points (same as lorenz.py)"""
def latin_hypercube(n: int, a: float, b: float) -> torch.Tensor:
    edges = torch.linspace(0, 1, n + 1)
    u = torch.rand(n)
    samples = edges[:-1] + u / n
    samples = samples[torch.randperm(n)]
    samples = a + (b - a) * samples
    return samples.view(-1, 1)

def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

"""Training loss: mean-squared Lorenz residual over the whole horizon [0, T]"""
def physics_loss(model, sigma, rho, beta, time_horizon, verbose=False):
    ts = latin_hypercube(1000, 0, time_horizon).requires_grad_(True)
    u = model(ts)
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    dx, dy, dz = grad(x, ts), grad(y, ts), grad(z, ts)
    eq1 = dx.squeeze() - sigma * (y - x)
    eq2 = dy.squeeze() - (x * (rho - z) - y)
    eq3 = dz.squeeze() - (x * y - beta * z)
    loss1, loss2, loss3 = (eq1**2).mean(), (eq2**2).mean(), (eq3**2).mean()
    if verbose:
        print(f"  loss1={loss1.item():.6e}  loss2={loss2.item():.6e}  loss3={loss3.item():.6e}")
    return loss1 + loss2 + loss3

class Net(nn.Module):
    def __init__(self, time_horizon, x0, y0, z0):
        super().__init__()
        self.time_horizon = time_horizon
        self.x0, self.y0, self.z0 = x0, y0, z0
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        tau = t / self.time_horizon
        result = self.linear_stack(tau)
        return torch.cat([
            self.x0 + t * 5 * result[:, 0:1],
            self.y0 + t * 5 *result[:, 1:2],
            self.z0 + t * 5 * result[:, 2:3],
        ], dim=1)

"""Lorenz residuals of the trained PINN on a grid"""
def residuals(model, sigma, rho, beta, t_grid):
    ts = t_grid.clone().detach().requires_grad_(True)
    u = model(ts)
    x = u[:, 0]
    y = u[:, 1]
    z = u[:, 2]
    dx, dy, dz = grad(x, ts), grad(y, ts), grad(z, ts)
    r1 = (dx.squeeze() - sigma * (y - x)).detach().numpy()
    r2 = (dy.squeeze() - (x * (rho - z) - y)).detach().numpy()
    r3 = (dz.squeeze() - (x * y - beta * z)).detach().numpy()
    return r1, r2, r3

if __name__ == "__main__":
    torch.manual_seed(0)
    lr = 1e-3
    sigma, rho, beta = 10, 28, 8 / 3
    x0, y0, z0 = 1.0, 0.2, 1.0
    z_star = rho - 1
    time_horizon = 0.4
    epochs = 50_000
    print_every = 5000
    t_lo, t_hi = 0.3, 0.4

    net = Net(time_horizon, x0, y0, z0)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        opt.zero_grad()
        loss = physics_loss(net, sigma, rho, beta, time_horizon, verbose=(epoch % print_every == 0))
        loss.backward()
        opt.step()
        if epoch % print_every == 0:
            print(f"epoch {epoch}  loss = {loss.item():.6e}")

    # residuals of the trained PINN over the full horizon [0, T]
    t_grid = torch.linspace(t_lo, t_hi, 2000).view(-1, 1)
    r1, r2, r3 = residuals(net, sigma, rho, beta, t_grid)
    tt = t_grid.numpy().flatten()
    total = np.sqrt(r1**2 + r2**2 + r3**2)

    # reference solution to locate z = rho-1 crossing(s) inside the window
    def lorenz(t, X, s, b, r):
        x, y, z = X
        return s * (y - x), x * (r - z) - y, x * y - b * z
    sol = solve_ivp(lorenz, (0, time_horizon), (x0, y0, z0), args=(sigma, beta, rho), dense_output=True, rtol=1e-12, atol=1e-12)
    tref = np.linspace(t_lo, t_hi, 10000)
    xref, yref, zref = sol.sol(tref)
    stable = zref > z_star
    idx = np.where(np.diff(stable.astype(int)) != 0)[0]
    crossings = tref[idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tt, r1, label=r"$r_x=\dot{x}-\sigma(y-x)$")
    ax.plot(tt, r2, label=r"$r_y=\dot{y}-(x(\rho-z)-y)$")
    ax.plot(tt, r3, label=r"$r_z=\dot{z}-(xy-\beta z)$")
    ax.plot(tt, total, "k--", label=r"$\|r\|$")
    # ax.axvspan(t_lo, t_hi, color="gray", alpha=0.15)
    for i, c in enumerate(crossings):
        ax.axvline(c, color="red", ls=":", label=(r"$z^\star$ crossing" if i == 0 else None))
    ax.set_xlabel("t")
    ax.set_ylabel("residual")
    ax.set_title(rf"Lorenz PINN residuals on $[{t_lo},{t_hi}]$ ({epochs} epochs)")
    ax.legend()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/lorenz_residuals_{t_lo}_{t_hi}.png", dpi=300, bbox_inches="tight")
    print("crossings in [0.3,0.4]:", crossings)
    print("saved figures/lorenz_residuals_th0.4.png")

    # # PINN trajectory vs reference solution over the full horizon
    # t_full = np.linspace(0, time_horizon, 2000)
    # with torch.no_grad():
    #     pred = net(torch.tensor(t_full, dtype=torch.float32).view(-1, 1)).numpy()
    # ref = np.stack(sol.sol(t_full), axis=1)
    # err = np.abs(pred - ref)
    # print(f"max |PINN - reference|: x={err[:, 0].max():.3e}  y={err[:, 1].max():.3e}  z={err[:, 2].max():.3e}")

    # fig2, ax2 = plt.subplots(figsize=(9, 5))
    # for j, name in enumerate(["x", "y", "z"]):
    #     ax2.plot(t_full, ref[:, j], "--", color=f"C{j}", label=f"Exact {name}")
    #     ax2.plot(t_full, pred[:, j], color=f"C{j}", label=f"PINN {name}")
    # ax2.axvspan(t_lo, t_hi, color="gray", alpha=0.15)
    # for c in crossings:
    #     ax2.axvline(c, color="red", ls=":")
    # ax2.set_xlabel("t")
    # ax2.set_ylabel("state")
    # ax2.set_title(rf"Lorenz PINN on $[0,{time_horizon}]$ ({epochs} epochs)")
    # ax2.legend()
    # plt.savefig(f"figures/lorenz_residuals_fit_th{time_horizon}.png", dpi=300, bbox_inches="tight")
    # print(f"saved figures/lorenz_residuals_fit_th{time_horizon}.png")