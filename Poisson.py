# ============================================================
# PINN for 2D Poisson on (0,1)x(0,1) with Dirichlet BC only
#   -Δu = f,  u=0 on boundary
# Manufactured solution:
#   u*(x,y) = sin(pi x) sin(pi y)
#   f(x,y)  = 2*pi^2*sin(pi x) sin(pi y)
# ============================================================

import math, time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt

# --------------------- device / seed ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
print("device:", device)

PI = math.pi

# --------------------- exact solution & RHS ---------------------
def u_true(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return torch.sin(PI * x) * torch.sin(PI * y)

def f_rhs(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return 2.0 * (PI**2) * torch.sin(PI * x) * torch.sin(PI * y)

# --------------------- MLP ---------------------
class MLP(nn.Module):
    def __init__(self, width=64, depth=6):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy):
        return self.net(xy)

model = MLP(width=64, depth=6).to(device)

# --------------------- autograd helpers ---------------------
def d(u, x):
    return grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]

def laplacian(u, xy):
    # u: [N,1], xy: [N,2] requires_grad=True
    gu = d(u, xy)               # [N,2] => (u_x, u_y)
    u_x = gu[:, 0:1]
    u_y = gu[:, 1:2]
    u_xx = d(u_x, xy)[:, 0:1]
    u_yy = d(u_y, xy)[:, 1:2]
    return u_xx + u_yy

# --------------------- sampling ---------------------
def sample_interior(N):
    xy = torch.rand(N, 2, device=device)
    xy.requires_grad_(True)
    return xy

def sample_boundary(N):
    # split across 4 edges
    N1 = N // 4
    N2 = N // 4
    N3 = N // 4
    N4 = N - (N1 + N2 + N3)

    y1 = torch.rand(N1, 1, device=device); x1 = torch.zeros_like(y1)       # x=0
    y2 = torch.rand(N2, 1, device=device); x2 = torch.ones_like(y2)        # x=1
    x3 = torch.rand(N3, 1, device=device); y3 = torch.zeros_like(x3)       # y=0
    x4 = torch.rand(N4, 1, device=device); y4 = torch.ones_like(x4)        # y=1

    xy = torch.cat([
        torch.cat([x1, y1], dim=1),
        torch.cat([x2, y2], dim=1),
        torch.cat([x3, y3], dim=1),
        torch.cat([x4, y4], dim=1),
    ], dim=0)
    # boundary loss doesn't need grads w.r.t. inputs, but harmless either way
    return xy

# --------------------- training settings ---------------------
N_f = 50_000     # interior collocation
N_b = 10_000     # boundary points
epochs = 20_000
print_every = 1000

lambda_bc = 100.0  # boundary weight (Dirichlet only)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=800, threshold=1e-6, min_lr=1e-6
)

# fixed training samples (you can also resample every epoch if you want)
xy_f = sample_interior(N_f)
xy_b = sample_boundary(N_b)

# --------------------- losses ---------------------
def losses():
    # PDE residual: -Δu - f = 0
    u_f = model(xy_f)
    lap_u = laplacian(u_f, xy_f)
    r = -(lap_u) - f_rhs(xy_f)
    L_pde = torch.mean(r**2)

    # Dirichlet BC: u=0 on boundary
    u_b = model(xy_b)
    L_bc = torch.mean(u_b**2)

    L = L_pde + lambda_bc * L_bc
    return L, L_pde, L_bc

# --------------------- train ---------------------
t0 = time.time()
best = float("inf")

for ep in range(1, epochs + 1):
    opt.zero_grad(set_to_none=True)
    L, Lpde, Lbc = losses()
    L.backward()
    opt.step()

    scheduler.step(L.item())

    if ep % print_every == 0 or ep == 1:
        lr = opt.param_groups[0]["lr"]
        print(f"[{ep:5d}] L={L.item():.3e}  PDE={Lpde.item():.3e}  BC={Lbc.item():.3e}  lr={lr:.2e}")

print(f"done in {time.time()-t0:.1f}s")

# --------------------- evaluation ---------------------
@torch.no_grad()
def eval_rel_l2(n=201):
    xs = torch.linspace(0, 1, n, device=device)
    ys = torch.linspace(0, 1, n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)

    up = model(xy).reshape(n, n)
    ut = u_true(xy).reshape(n, n)

    num = torch.sqrt(torch.mean((up - ut)**2))
    den = torch.sqrt(torch.mean((ut)**2)) + 1e-12
    return (num / den).item(), up.cpu().numpy(), ut.cpu().numpy()

rel_l2, U_pred, U_true_np = eval_rel_l2(n=200)
print(f"[Eval] relative L2 (RMS) error = {rel_l2:.3e}")

# residual heatmap (needs grads)
def eval_residual(n=200):
    xs = torch.linspace(0, 1, n, device=device)
    ys = torch.linspace(0, 1, n, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="ij")
    xy = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1).detach().requires_grad_(True)

    u = model(xy)
    r = -(laplacian(u, xy)) - f_rhs(xy)
    return r.detach().reshape(n, n).cpu().numpy()

R = eval_residual(n=200)

# --------------------- plots ---------------------
fig = plt.figure(figsize=(13, 3.8))

ax1 = plt.subplot(1, 3, 1)
im1 = ax1.imshow(U_true_np, origin="lower", extent=[0,1,0,1], aspect="auto")
ax1.set_title("True u*(x,y)")
ax1.set_xlabel("y"); ax1.set_ylabel("x")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

ax2 = plt.subplot(1, 3, 2)
im2 = ax2.imshow(U_pred, origin="lower", extent=[0,1,0,1], aspect="auto")
ax2.set_title("PINN u(x,y)")
ax2.set_xlabel("y"); ax2.set_ylabel("x")
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

ax3 = plt.subplot(1, 3, 3)
im3 = ax3.imshow(np.abs(U_pred - U_true_np), origin="lower", extent=[0,1,0,1], aspect="auto")
ax3.set_title(f"|error|  (rel L2={rel_l2:.2e})")
ax3.set_xlabel("y"); ax3.set_ylabel("x")
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

plt.figure(figsize=(5.2, 4.2))
im = plt.imshow(R, origin="lower", extent=[0,1,0,1], aspect="auto")
plt.title("Residual r(x,y) = -Δu - f")
plt.xlabel("y"); plt.ylabel("x")
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()