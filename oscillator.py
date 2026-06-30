import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

m = 1.5
gamma = 0
k = 1.5
epochs = 1001

u_0 = -2.0
du_0 = 0.0

time_horizon = 8

def latin_hypercube(n, a, b):
    edges = torch.linspace(0, 1, n + 1)

    u = torch.rand(n)

    samples = edges[:-1] + u / n

    samples = samples[torch.randperm(n)]

    samples = a + (b - a) * samples

    return samples.view(-1, 1)

ts_train = latin_hypercube(68, 0, time_horizon).requires_grad_(True)

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs, 
        inputs, 
        grad_outputs=torch.ones_like(outputs), 
        create_graph=True
    )[0]

def physics_loss(model):
    # ts = (torch.rand(68, 1) * time_horizon).view(-1, 1).requires_grad_(True)

    # x = torch.rand(3000,1)
    # ts = (time_horizon * (1 - torch.sqrt(1-x)).view(-1, 1)).requires_grad_(True)
    
    u = model(ts_train)
    ux = grad(u, ts_train)
    uxx = grad(ux, ts_train)
    
    ode = m * uxx + gamma * ux + k * u

    return torch.mean(ode**2)

def ic_loss(model):
    t0 = torch.tensor([[0.0]], requires_grad=True)

    u0 = model(t0)
    du0 = grad(u0, t0)

    loss_u = (u0 - u_0)**2
    loss_du = (du0 - du_0)**2

    return loss_u + loss_du

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(1, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1)
        )
    
    def forward(self, x):
        x = x / time_horizon
        return self.linear_stack(x)
    
    
    def predict(self, X):
        self.eval()
        out = self.forward(X)
        return out.detach().cpu().numpy()

    
net = Net()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

X = torch.linspace(0, time_horizon, 1000).view(-1, 1)
x = X.numpy().flatten()

for epoch in range(epochs):
    optimizer.zero_grad()
    
    loss_phys = physics_loss(net)
    loss_ic = ic_loss(net)
    loss = loss_phys + loss_ic

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(epoch, loss.item())
        yp = net.predict(X).flatten()
        plt.plot(x, yp, label=f'PINN at epoch {epoch}')
        print("physics = ", loss_phys.item(),
              "IC = ", loss_ic.item()
        )

# y = np.exp(-0.1*x) * (
#     np.cos(np.sqrt(0.99)*x)
#     + (0.1/np.sqrt(0.99))*np.sin(np.sqrt(0.99)*x)
# )

y = -2 * np.cos(x)

plt.plot(x, y, "--", label="Exact")
plt.legend()
plt.show()