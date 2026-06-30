import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sigma = 10
rho = 28
beta = 8/3
time_horizon = 0.2

x0 = 1.0
y0 = 1.0
z0 = 1.0

epochs = 5000

def latin_hypercube(n, a, b):
    edges = torch.linspace(0, 1, n + 1)

    u = torch.rand(n)

    samples = edges[:-1] + u / n

    samples = samples[torch.randperm(n)]

    samples = a + (b - a) * samples

    return samples.view(-1, 1)

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, 
        inputs, 
        grad_outputs=torch.ones_like(outputs), 
        create_graph=True
    )[0]

def physics_loss(model):
    ts = latin_hypercube(3000, 0, time_horizon).requires_grad_(True)

    # x = torch.rand(3000,1)
    # ts = (30 * (1 - torch.sqrt(1-x)).view(-1, 1)).requires_grad_(True)
    
    u = model(ts)

    x = u[:, 0]
    y = u[:, 1]
    z = u[:, 2]

    dx = grad(x, ts)
    dy = grad(y, ts)
    dz = grad(z, ts)
    
    eq1 = dx.squeeze() - sigma * (y - x)
    eq2 = dy.squeeze() - (x * (rho - z) - y)
    eq3 = dz.squeeze() - (x * y - beta * z)

    loss1 = (eq1**2).mean()
    loss2 = (eq2**2).mean()
    loss3 = (eq3**2).mean()

    print(loss1.item(), loss2.item(), loss3.item())

    return 1.0 * loss1 + 1.0 * loss2 + 0.1 * loss3

def ic_loss(model):
    t0 = torch.tensor([[0.0]], requires_grad=True)

    u0 = model(t0)

    # IC: u(0, 0, 0) = (0, 0.5, 0)

    loss_x0 = (u0[0, 0] - x0)**2
    loss_y0 = (u0[0, 1] - y0)**2
    loss_z0 = (u0[0, 2] - z0)**2

    # loss_u = (u0 - 1.0)**2
    # loss_du = (du0 - 0.0)**2

    return loss_x0 + loss_y0 + loss_z0

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )
    
    def forward(self, t):
        tau = t / time_horizon
        result = self.linear_stack(tau)
        return torch.cat([
            x0 + t * result[:, 0:1],
            y0 + t * result[:, 1:2],
            z0 + t * result[:, 2:3],
        ], dim=1)
    
    
    def predict(self, X):
        self.eval()
        out = self.forward(X)
        return out.detach().cpu().numpy()

    
net = Net()

for m in net.modules():
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

T = torch.linspace(0, time_horizon, 1000).view(-1, 1)
t = T.numpy().flatten()

for epoch in range(epochs + 1):
    optimizer.zero_grad()
    
    loss_phys = physics_loss(net)
    # loss_ic = ic_loss(net)
    # loss = loss_phys + 100 * loss_ic
    loss = loss_phys

    loss.backward()

    optimizer.step()

    if epoch % 5000 == 0:
        print(epoch, loss.item())
        xp = net.predict(T)[:, 0]
        yp = net.predict(T)[:, 1]
        zp = net.predict(T)[:, 2]
        plt.plot(t, xp, label=f'PINN x_p at epoch {epoch}')
        plt.plot(t, yp, label=f'PINN y_p at epoch {epoch}')
        plt.plot(t, zp, label=f'PINN z_p at epoch {epoch}')
        print("physics = ", loss_phys.item(),
            #   "IC = ", loss_ic.item()
        )

# y = np.exp(-0.1*x) * (
#     np.cos(np.sqrt(0.99)*x)
#     + (0.1/np.sqrt(0.99))*np.sin(np.sqrt(0.99)*x)
# )

# plt.plot(t, y, "--", label="Exact")

# numerically integrated solution

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

plt.legend()
plt.show()