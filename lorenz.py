import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
plt.rcParams['text.usetex'] = (sys.platform == "darwin")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = ['Computer Modern']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath, amssymb}'
from scipy.integrate import solve_ivp

"""Sampling function for generating training data withing the time horizon"""
def latin_hypercube(n: int, a: float, b: float) -> torch.Tensor:
    edges = torch.linspace(0, 1, n + 1)
    u = torch.rand(n)
    samples = edges[:-1] + u / n
    samples = samples[torch.randperm(n)]
    samples = a + (b - a) * samples
    return samples.view(-1, 1)

"""Gradient function for computing gradients"""
def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True
    )[0]

"""Loss function computing the residuals of the Lorenz system equations"""
def physics_loss(model: nn.Module, sigma: float, rho: float, beta: float, time_horizon: float, verbose: bool = False) -> torch.Tensor:
    ts = latin_hypercube(3000, 0, time_horizon).requires_grad_(True)
    u = model(ts)

    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    dx, dy, dz = grad(x, ts), grad(y, ts), grad(z, ts)

    eq1 = dx.squeeze() - sigma * (y - x)
    eq2 = dy.squeeze() - (x * (rho - z) - y)
    eq3 = dz.squeeze() - (x * y - beta * z)

    loss1, loss2, loss3 = (eq1**2).mean(), (eq2**2).mean(), (eq3**2).mean()

    if verbose:
        print(f"  loss1={loss1.item():.6e}  loss2={loss2.item():.6e}  loss3={loss3.item():.6e}")
    return 1.0 * loss1 + 1.0 * loss2 + 1.0 * loss3

"""Neural network architecture for the PINN"""
class Net(nn.Module):
    def __init__(self, time_horizon: float, x0: float, y0: float, z0: float) -> None:
        super().__init__()
        self.time_horizon = time_horizon
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

    """Forward pass; tau is the normalized time input, output is the hard-constrained state (x, y, z)"""
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        tau = t / self.time_horizon
        result = self.linear_stack(tau)
        return torch.cat([
            self.x0 + t * result[:, 0:1],
            self.y0 + t * result[:, 1:2],
            self.z0 + t * result[:, 2:3],
        ], dim=1)

    """Predict the state variables with the model in evaluation mode"""
    def predict(self, X: torch.Tensor) -> np.ndarray:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self.forward(X)
        if was_training:
            self.train()
        return out.cpu().numpy()

if __name__ == "__main__":
    lr = 1e-3
    sigma = 10
    rho = 28
    beta = 8/3
    time_horizon = 0.3
    x0 = 1.0
    y0 = 1.0
    z0 = 1.0
    epochs = 5000
    print_every = 500

    net = Net(time_horizon, x0, y0, z0)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    T = torch.linspace(0, time_horizon, 1000).view(-1, 1)
    t = T.numpy().flatten()

    for epoch in range(epochs + 1):
        optimizer.zero_grad()

        verbose = epoch % print_every == 0
        loss_phys = physics_loss(net, sigma, rho, beta, time_horizon, verbose=verbose)
        loss = loss_phys

        loss.backward()
        optimizer.step()

        if verbose:
            print(f"epoch {epoch}  loss = {loss.item():.6e}")
        if epoch % 5000 == 0:
            xp = net.predict(T)[:, 0]
            yp = net.predict(T)[:, 1]
            zp = net.predict(T)[:, 2]
            plt.plot(t, xp, label=f'PINN x_p at epoch {epoch}')
            plt.plot(t, yp, label=f'PINN y_p at epoch {epoch}')
            plt.plot(t, zp, label=f'PINN z_p at epoch {epoch}')

    """Reference solution via direct numerical integration of the Lorenz system"""
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

    plt.plot(t, x, "--", label="Exact x")
    plt.plot(t, y, "--", label="Exact y")
    plt.plot(t, z, "--", label="Exact z")
    plt.xlabel("t")
    plt.ylabel("state")
    plt.title(f"Lorenz attractor PINN fit (time horizon = {time_horizon})")
    plt.legend()

    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/lorenz_fit_th{time_horizon}.png", dpi=300, bbox_inches="tight")
    plt.show()