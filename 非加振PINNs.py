# ============================================================
# Conventional PINN for 1D Shallow Water (no forcing)
#   + Hard IC: value AND time-derivative fixed (η, u)
#   + Weak Neumann (η_x=0) at walls (weight adjustable; default 0)
#   + Time-collocation stratified sampling (100 bins)
#   + RAR at 25%, 50%, 75% epochs: add 20000 points each (Nx=512,Nt=64)
#   + LR scheduler: ReduceLROnPlateau (patience=800)
#   + Output for thesis-ready results (plots + npz)
#     - training curves
#     - time-series errors vs linear solution (η,u)
#     - wall-violation vs time
#     - mass condition (mean η) vs time
#     - snapshots (η,u profiles) at selected times
#     - overlay GIF (η) and optional u GIF
# ============================================================

import math, time, io, os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ===================== USER SWITCH =====================
BC_MODE = "hard"   # "soft" or "hard"
# =======================================================

# ------------------- Output dir -------------------
OUTDIR = "/content/Downloads"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------- Physics & benchmark ----------------
g   = 9.81
Lx  = 2.0
h0  = 1.0
n_mode = 1
k   = n_mode * math.pi / Lx
c   = math.sqrt(g * h0)
omega = k * c

A   = 0.01
phi = 0.0

T1 = 2.0 * math.pi / omega
T  = 2.0 * T1  # 3周期

# ---------------- PINN hyperparams ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

N_f   = 20000
N_ic  = 2000
N_bc  = 2000
M_t   = 64
M_x   = 256

E_adam  = 6_000
E_lbfgs = True

# loss weights
w_fm, w_fp       = 1.0, 1.0
w_ic_eta, w_ic_u = 1.0, 1.0
w_bc_u           = 100.0
w_mass           = 5.0

# ---------------- Stratified sampling settings ----------------
N_STRATA_T = 100  # time bins

# ---------------- RAR settings ----------------
RAR_ADD_K = 20000
RAR_NX = 512
RAR_NT = 64
RAR_EPOCHS = [int(0.25*E_adam), int(0.50*E_adam), int(0.75*E_adam)]

# ---------------- LR scheduler settings ----------------
SCHED_PATIENCE = 800

# ---------------- Plot / eval settings ----------------
LOG_EVERY = 50           # store training log every N epochs
PRINT_EVERY = 500
EVAL_NT = 200            # time samples for evaluation curves
EVAL_NX = 401            # space samples for evaluation curves
SNAP_TIMES = [0.0, 0.25*T, 0.50*T, 0.75*T, 1.00*T]  # snapshot times
MAKE_U_GIF = False       # if True, also make u overlay GIF

# ---------------- Utilities ----------------
def d(y, x):
    return grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]

def linear_eta_u(x, t):
    # Linear standing-wave (benchmark)
    return (
        A*torch.cos(k*x)*torch.cos(omega*t + phi),
        (A*math.sqrt(g*h0)/h0)*torch.sin(k*x)*torch.sin(omega*t + phi)
    )

@torch.no_grad()
def linear_eta_u_torch(x, t):
    return (
        A*torch.cos(k*x)*torch.cos(omega*t + phi),
        (A*math.sqrt(g*h0)/h0)*torch.sin(k*x)*torch.sin(omega*t + phi)
    )

def stratified_time_samples(N, t0, t1, n_bins=100, device="cpu", dtype=torch.float32):
    t0 = float(t0); t1 = float(t1)
    n_bins = int(n_bins)
    assert n_bins >= 1
    N = int(N)

    base = N // n_bins
    rem  = N - base*n_bins
    counts = [base + (1 if i < rem else 0) for i in range(n_bins)]

    edges = torch.linspace(t0, t1, n_bins+1, device=device, dtype=dtype)
    ts = []
    for i in range(n_bins):
        ni = counts[i]
        if ni <= 0:
            continue
        a = edges[i]
        b = edges[i+1]
        u = torch.rand(ni, 1, device=device, dtype=dtype)
        ts.append(a + (b - a) * u)

    t = torch.cat(ts, dim=0)
    idx = torch.randperm(t.shape[0], device=device)
    return t[idx]

# ---------------- Normalized exponential^2 gate ----------------
def s_exp2_norm(t, T, a):
    """
    s(t) = ((1 - exp(-a t)) / (1 - exp(-a T)))^2
    -> s(0)=0, s'(0)=0, s(T)=1
    """
    Tt = torch.as_tensor(T, device=t.device, dtype=t.dtype)
    at = torch.as_tensor(a, device=t.device, dtype=t.dtype)

    gT = 1.0 - torch.exp(-at * Tt)
    gT = torch.clamp(gT, min=1e-8)

    gt = 1.0 - torch.exp(-at * t)
    return (gt / gT)**2

# ---------------- Network ----------------
class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, width=40, depth=8, bc_mode="soft"):
        super().__init__()
        self.bc_mode = bc_mode

        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        self.a_eta = 15.0 / T
        self.a_u   = 15.0 / T

    def forward(self, x, t):
        x_nd = x / Lx
        t_nd = t / T

        # exact IC values at t=0
        eta0, u0 = linear_eta_u(x, torch.zeros_like(t))

        # exact IC time-derivatives at t=0 (from linear solution)
        # phi is constant; keep as tensor for device/dtype safety
        phi_t = torch.as_tensor(phi, device=x.device, dtype=x.dtype)
        eta_t0 = -A * omega * torch.cos(k*x) * torch.sin(phi_t)
        u_t0   = (A*math.sqrt(g*h0)/h0) * omega * torch.sin(k*x) * torch.cos(phi_t)

        out = self.net(torch.cat([x_nd, t_nd], dim=1))
        eta_hat = out[:, :1]
        u_hat   = out[:, 1:2] * c  # physical scale

        s_eta = s_exp2_norm(t, T, self.a_eta)
        s_u   = s_exp2_norm(t, T, self.a_u)

        # Hard IC with value + time-derivative fixed
        eta = eta0 + t*eta_t0 + s_eta * eta_hat

        u_core = u0 + t*u_t0 + s_u * u_hat

        # ---- BC switch (only this part differs) ----
        if self.bc_mode == "hard":
            # b(0)=0, b(L)=0  via sin(pi * x/L)
            b = torch.sin(math.pi * x_nd)
            u = b * u_core
        else:
            # soft: enforce via BC loss
            u = u_core
        # -------------------------------------------

        return eta, u

model = MLP(bc_mode=BC_MODE).to(device)

# Optimizer (Adam)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# LR scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=SCHED_PATIENCE, threshold=1e-6, min_lr=1e-6
)

# ---------------- Sample training points ----------------
# collocation: stratified in time + uniform x
t_f = stratified_time_samples(
    N_f, 0.0, T, n_bins=N_STRATA_T, device=device, dtype=torch.get_default_dtype()
)
x_f = (torch.rand(N_f, 1, device=device) * Lx)

# make leaf tensors with grad
x_f = x_f.detach().requires_grad_(True)
t_f = t_f.detach().requires_grad_(True)

# IC points (diagnostic)
x_ic = (torch.rand(N_ic, 1, device=device) * Lx)
t_ic = torch.zeros_like(x_ic, device=device)
with torch.no_grad():
    eta0_t, u0_t = linear_eta_u(x_ic, t_ic)

# BC Dirichlet u=0 on walls
t_bcL = (torch.rand(N_bc//2, 1, device=device) * T)
x_bcL = torch.zeros_like(t_bcL, device=device)
t_bcR = (torch.rand(N_bc - N_bc//2, 1, device=device) * T)
x_bcR = torch.full_like(t_bcR, Lx, device=device)

# Mass times for constraint
t_mass = torch.linspace(0.0, T, M_t, device=device).reshape(-1,1)

# ---------------- PDE residual helper (for RAR) ----------------
def pde_residuals(x, t):
    eta, u = model(x, t)
    h = h0 + eta
    q = h * u
    r_mass = d(h, t) + d(q, x)
    r_mom  = d(q, t) + d(h*u*u + 0.5*g*h*h, x)
    return r_mass, r_mom

# ---------------- RAR step ----------------
def rar_step(add_k=20000, Nx=512, Nt=64):
    global x_f, t_f

    xc = torch.linspace(0.0, Lx, Nx, device=device, dtype=torch.get_default_dtype()).reshape(-1,1)
    tc = torch.linspace(0.0, T,  Nt, device=device, dtype=torch.get_default_dtype()).reshape(-1,1)
    X, TT = torch.meshgrid(xc.squeeze(1), tc.squeeze(1), indexing="ij")
    X  = X.reshape(-1,1)
    TT = TT.reshape(-1,1)

    X  = X.detach().requires_grad_(True)
    TT = TT.detach().requires_grad_(True)

    r_m, r_p = pde_residuals(X, TT)
    score = (r_m**2 + r_p**2).detach().reshape(-1)

    k_take = min(int(add_k), score.numel())
    topk = torch.topk(score, k_take, largest=True, sorted=False).indices

    x_new = X.detach()[topk].reshape(-1,1)
    t_new = TT.detach()[topk].reshape(-1,1)

    x_f = torch.cat([x_f.detach(), x_new], dim=0).detach().requires_grad_(True)
    t_f = torch.cat([t_f.detach(), t_new], dim=0).detach().requires_grad_(True)

    print(f"[RAR] added {k_take} points. collocation now: {x_f.shape[0]}")

# ---------------- Loss (nonlinear SWE, no forcing) -------------
def losses():
    # --- PDE residuals (2) ---
    eta_f, u_f = model(x_f, t_f)
    h_f = h0 + eta_f
    q_f = h_f * u_f

    r_mass = d(h_f, t_f) + d(q_f, x_f)
    r_mom  = d(q_f, t_f) + d(h_f*u_f*u_f + 0.5*g*h_f*h_f, x_f)

    L_fm = torch.mean(r_mass**2)
    L_fp = torch.mean(r_mom**2)

    # --- IC diagnostics (2) ---
    eta_i, u_i = model(x_ic, t_ic)
    L_ic_eta = torch.mean((eta_i - eta0_t)**2)
    L_ic_u   = torch.mean((u_i   - u0_t  )**2)

    # --- BC u=0 (soft only) ---
    if BC_MODE == "soft":
        _, u_L = model(x_bcL, t_bcL)
        _, u_R = model(x_bcR, t_bcR)
        L_bc_u = 0.5*(torch.mean(u_L**2) + torch.mean(u_R**2))
    else:
    # hard: do not compute BC loss (same behavior as forced code)
        L_bc_u = torch.zeros((), device=device)

    # --- Mass constraint mean(η)=0 (1) ---
    L_mass = 0.0
    for tk in t_mass:
        x_s = torch.linspace(0.0, Lx, M_x, device=device).reshape(-1,1)
        t_s = torch.full_like(x_s, float(tk.item()))
        eta_s, _ = model(x_s, t_s)
        L_mass = L_mass + (torch.mean(eta_s))**2
    L_mass = L_mass / len(t_mass)

    total = (w_fm*L_fm + w_fp*L_fp +
             w_ic_eta*L_ic_eta + w_ic_u*L_ic_u +
             w_bc_u*L_bc_u +
             w_mass*L_mass)

    return total, {
        "fm":L_fm, "fp":L_fp,
        "ic_e":L_ic_eta, "ic_u":L_ic_u,
        "bc_u":L_bc_u,
        "mass":L_mass
    }

# ---------------- Train ----------------
history = {
    "epoch": [],
    "lr": [],
    "L_tot": [],
    "L_fm": [], "L_fp": [],
    "L_ic_eta": [], "L_ic_u": [],
    "L_bc_u": [],
    "L_mass": [],
    "Nf": [],
}

t0_train = time.time()
for ep in range(1, E_adam+1):
    if ep in RAR_EPOCHS:
        print(f"[RAR] trigger at epoch {ep} ...")
        rar_step(add_k=RAR_ADD_K, Nx=RAR_NX, Nt=RAR_NT)

    opt.zero_grad(set_to_none=True)
    Ltot, dct = losses()
    Ltot.backward()
    opt.step()

    lr_prev = opt.param_groups[0]["lr"]
    scheduler.step(Ltot.item())
    lr_now = opt.param_groups[0]["lr"]
    if lr_now < lr_prev:
        print(f"[LR] {lr_prev:.2e} -> {lr_now:.2e}")

    if ep % LOG_EVERY == 0 or ep == 1:
        history["epoch"].append(ep)
        history["lr"].append(lr_now)
        history["L_tot"].append(float(Ltot.item()))
        history["L_fm"].append(float(dct["fm"].item()))
        history["L_fp"].append(float(dct["fp"].item()))
        history["L_ic_eta"].append(float(dct["ic_e"].item()))
        history["L_ic_u"].append(float(dct["ic_u"].item()))
        history["L_bc_u"].append(float(dct["bc_u"].item()))
        history["L_mass"].append(float(dct["mass"].item()))
        history["Nf"].append(int(x_f.shape[0]))

    if ep % PRINT_EVERY == 0 or ep == 1:
        print(f"[Adam {ep:4d}] tot={Ltot.item():.3e} "
              f"fm={dct['fm'].item():.2e} fp={dct['fp'].item():.2e} "
              f"IC(η/u)=({dct['ic_e'].item():.2e}/{dct['ic_u'].item():.2e}) "
              f"BC(u)=({dct['bc_u'].item():.2e}) "
              f"mass={dct['mass'].item():.2e} "
              f"lr={lr_now:.2e} "
              f"Nf={x_f.shape[0]}")

if E_lbfgs:
    lb = torch.optim.LBFGS(
        model.parameters(), lr=1.0, max_iter=500,
        tolerance_grad=1e-8, tolerance_change=1e-9,
        history_size=50, line_search_fn="strong_wolfe"
    )
    def closure():
        lb.zero_grad(set_to_none=True)
        Ltot,_ = losses()
        Ltot.backward()
        return Ltot
    print("[LBFGS] start"); lb.step(closure); print("[LBFGS] done")

print(f"Training done in {time.time()-t0_train:.1f}s")

# ---------------- Save training curves ----------------
hist_path = os.path.join(OUTDIR, "nonforced_pinn_training_history.npz")
np.savez(hist_path, **{k: np.array(v) for k, v in history.items()})
print("Saved:", hist_path)

# Plot training curves
fig = plt.figure(figsize=(10.2, 6.2))
ep = np.array(history["epoch"])

plt.semilogy(ep, history["L_tot"], label="L_total")
plt.semilogy(ep, history["L_fm"],  label="L_mass_PDE")
plt.semilogy(ep, history["L_fp"],  label="L_mom_PDE")

# ★ 加振PINNsと同じ：常に L_bc_u を描画（Soft/Hard両方）
plt.semilogy(ep, history["L_bc_u"], label="L_bc_u (soft only)")

plt.semilogy(ep, history["L_mass"], label="L_mean_eta")
plt.semilogy(ep, np.array(history["L_ic_eta"]) + np.array(history["L_ic_u"]), label="L_IC_diag (eta+u)")

plt.xlabel("epoch"); plt.ylabel("loss (log scale)")
plt.title(f"Training curves (non-forced SWE PINN)  BC_MODE={BC_MODE}")
plt.grid(True, alpha=0.3); plt.legend()

train_png = os.path.join(OUTDIR, "nonforced_training_curves.png")
plt.tight_layout(); plt.savefig(train_png, dpi=180); plt.close(fig)
print("Saved:", train_png)

# ---------------- Evaluation vs linear solution ----------------
@torch.no_grad()
def eval_time_series(nx=EVAL_NX, nt=EVAL_NT):
    xs = torch.linspace(0.0, Lx, nx, device=device).reshape(-1,1)
    ts = torch.linspace(0.0, T, nt, device=device).reshape(-1,1)

    # normalization amplitudes for relative errors
    A_eta = max(A, 1e-12)
    A_u   = (A*math.sqrt(g*h0)/h0) + 1e-12

    e_eta = np.zeros(nt, dtype=float)
    e_u   = np.zeros(nt, dtype=float)

    mean_eta = np.zeros(nt, dtype=float)        # mass condition diagnostic
    uwall_L  = np.zeros(nt, dtype=float)        # wall violation
    uwall_R  = np.zeros(nt, dtype=float)

    x0 = torch.zeros(1,1, device=device)
    xL = torch.full((1,1), Lx, device=device)

    for i, tk in enumerate(ts):
        tcur = float(tk.item())
        tcol = torch.full_like(xs, tcur)

        eta_p, u_p = model(xs, tcol)
        eta_l, u_l = linear_eta_u_torch(xs, tcol)

        e_eta[i] = float(torch.sqrt(torch.mean((eta_p-eta_l)**2)).cpu()) / A_eta
        e_u[i]   = float(torch.sqrt(torch.mean((u_p-u_l)**2)).cpu()) / A_u

        mean_eta[i] = float(torch.mean(eta_p).cpu())

        t1 = torch.full((1,1), tcur, device=device)
        _, u0 = model(x0, t1)
        _, u1 = model(xL, t1)
        uwall_L[i] = float(torch.abs(u0).cpu())
        uwall_R[i] = float(torch.abs(u1).cpu())

    return (
        np.linspace(0.0, T, nt),
        e_eta, e_u,
        mean_eta, uwall_L, uwall_R
    )

t_arr, e_eta_t, e_u_t, mean_eta_t, uwL_t, uwR_t = eval_time_series()

print(f"[Non-forced vs linear | BC_MODE={BC_MODE}] mean rel-RMS: eta={e_eta_t.mean():.3e}, u={e_u_t.mean():.3e}")
print(f"[Non-forced vs linear | BC_MODE={BC_MODE}] max  rel-RMS: eta={e_eta_t.max():.3e}, u={e_u_t.max():.3e}")
print(f"[Wall violation | BC_MODE={BC_MODE}] max |u(0,t)|={uwL_t.max():.3e}, max |u(L,t)|={uwR_t.max():.3e}")
print(f"[Mass condition | BC_MODE={BC_MODE}] max |mean_eta(t)|={np.max(np.abs(mean_eta_t)):.3e}")

eval_path = os.path.join(OUTDIR, "nonforced_eval_timeseries.npz")
np.savez(
    eval_path,
    t=t_arr, e_eta=e_eta_t, e_u=e_u_t,
    mean_eta=mean_eta_t, uwall_L=uwL_t, uwall_R=uwR_t
)
print("Saved:", eval_path)

# Plot time-series diagnostics
fig = plt.figure(figsize=(10.5, 7.2))
ax1 = plt.subplot(3,1,1)
ax1.plot(t_arr, e_eta_t, label="rel-RMS error (eta)")
ax1.plot(t_arr, e_u_t,   label="rel-RMS error (u)")
ax1.set_ylabel("relative RMS")
ax1.grid(True, alpha=0.3); ax1.legend()

ax2 = plt.subplot(3,1,2)
ax2.plot(t_arr, mean_eta_t)
ax2.set_ylabel("mean eta(t)")
ax2.set_title("Mass condition diagnostic: mean(eta) ~ 0")
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(3,1,3)
ax3.plot(t_arr, uwL_t, label="|u(0,t)|")
ax3.plot(t_arr, uwR_t, label="|u(L,t)|")
ax3.set_xlabel("t [s]"); ax3.set_ylabel("wall |u|")
ax3.grid(True, alpha=0.3); ax3.legend()

diag_png = os.path.join(OUTDIR, "nonforced_diagnostics_timeseries.png")
plt.tight_layout(); plt.savefig(diag_png, dpi=180); plt.close(fig)
print("Saved:", diag_png)

# ---------------- Snapshot profiles (PINN vs linear) ----------------
@torch.no_grad()
def plot_snapshots(times, nx=801):
    xs = torch.linspace(0.0, Lx, nx, device=device).reshape(-1,1)
    xnp = xs.squeeze().cpu().numpy()

    # eta snapshots
    fig1 = plt.figure(figsize=(10.6, 4.8))
    for tcur in times:
        tt = torch.full_like(xs, float(tcur))
        eta_p, _ = model(xs, tt)
        eta_l, _ = linear_eta_u_torch(xs, tt)
        plt.plot(xnp, eta_p.squeeze().cpu().numpy(), lw=2, label=f"PINN t={tcur:.2f}")
        plt.plot(xnp, eta_l.squeeze().cpu().numpy(), lw=2, ls="--", label=f"Linear t={tcur:.2f}")
    plt.axhline(0.0, ls=":", lw=1)
    plt.xlabel("x [m]"); plt.ylabel("eta(x,t) [m]")
    plt.title(f"Snapshots: eta(x,t) PINN vs Linear (non-forced)  BC_MODE={BC_MODE}")
    plt.grid(True, alpha=0.3); plt.legend(ncol=2, fontsize=9)
    eta_png = os.path.join(OUTDIR, "nonforced_snapshots_eta.png")
    plt.tight_layout(); plt.savefig(eta_png, dpi=180); plt.close(fig1)
    print("Saved:", eta_png)

    # u snapshots
    fig2 = plt.figure(figsize=(10.6, 4.8))
    for tcur in times:
        tt = torch.full_like(xs, float(tcur))
        _, u_p = model(xs, tt)
        _, u_l = linear_eta_u_torch(xs, tt)
        plt.plot(xnp, u_p.squeeze().cpu().numpy(), lw=2, label=f"PINN t={tcur:.2f}")
        plt.plot(xnp, u_l.squeeze().cpu().numpy(), lw=2, ls="--", label=f"Linear t={tcur:.2f}")
    plt.axhline(0.0, ls=":", lw=1)
    plt.xlabel("x [m]"); plt.ylabel("u(x,t) [m/s]")
    plt.title(f"Snapshots: u(x,t) PINN vs Linear (non-forced)  BC_MODE={BC_MODE}")
    plt.grid(True, alpha=0.3); plt.legend(ncol=2, fontsize=9)
    u_png = os.path.join(OUTDIR, "nonforced_snapshots_u.png")
    plt.tight_layout(); plt.savefig(u_png, dpi=180); plt.close(fig2)
    print("Saved:", u_png)

plot_snapshots(SNAP_TIMES)

# ---------------- Visualization GIF (overlay) ----------------
@torch.no_grad()
def make_gif_overlay(kind="eta",
                     out=None, n_frames=120, nx_plot=400, margin=0.25, dpi=170):
    """
    kind: "eta" or "u"
    """
    if out is None:
        out = os.path.join(OUTDIR, f"nonforced_pinn_vs_linear_{kind}.gif")

    xs_plot = torch.linspace(0.0, Lx, nx_plot, device=device).reshape(-1,1)
    xx = np.linspace(0.0, Lx, nx_plot)

    ts = np.linspace(0.0, T, n_frames)

    # auto y-lim
    amp = 0.0
    for tj in ts:
        t_t = torch.full_like(xs_plot, float(tj))
        eta_p, u_p = model(xs_plot, t_t)
        eta_l, u_l = linear_eta_u_torch(xs_plot, t_t)
        if kind == "eta":
            amp = max(amp, float(torch.max(torch.abs(eta_p)).cpu()), float(torch.max(torch.abs(eta_l)).cpu()))
        else:
            amp = max(amp, float(torch.max(torch.abs(u_p)).cpu()), float(torch.max(torch.abs(u_l)).cpu()))
    y_lim = (1.0 + margin) * max(amp, 1e-8)

    frames = []
    for tj in ts:
        t_t = torch.full_like(xs_plot, float(tj))
        eta_p, u_p = model(xs_plot, t_t)
        eta_l, u_l = linear_eta_u_torch(xs_plot, t_t)

        if kind == "eta":
            P = eta_p.squeeze().cpu().numpy()
            L = eta_l.squeeze().cpu().numpy()
            ylab = "eta(x,t) [m]"
            title = f"eta overlay  t={tj:.2f} s"
        else:
            P = u_p.squeeze().cpu().numpy()
            L = u_l.squeeze().cpu().numpy()
            ylab = "u(x,t) [m/s]"
            title = f"u overlay  t={tj:.2f} s"

        mse_x = float(np.mean((P - L)**2))

        fig = plt.figure(figsize=(6.9,3.6))
        plt.plot(xx, P, lw=2.2, label=f"PINN {kind}")
        plt.plot(xx, L, lw=2.2, ls="--", label=f"Linear {kind}")
        plt.axhline(0.0, ls=":", lw=1)
        plt.ylim(-y_lim, y_lim); plt.xlim(0.0, Lx)
        plt.xlabel("x [m]"); plt.ylabel(ylab)
        plt.title(f"{title}   MSE_x≈{mse_x:.2e}   BC_MODE={BC_MODE}")
        plt.grid(True, alpha=0.3); plt.legend(loc="upper right")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        import PIL.Image as Image
        frames.append(np.array(Image.open(buf).convert("RGB")))
        buf.close()

    imageio.mimsave(out, frames, duration=max(T/n_frames, 0.03))
    print("Saved GIF:", out)
    return out

gif_eta = make_gif_overlay(kind="eta", out=os.path.join(OUTDIR, "nonforced_pinn_vs_linear_eta.gif"))
if MAKE_U_GIF:
    gif_u = make_gif_overlay(kind="u", out=os.path.join(OUTDIR, "nonforced_pinn_vs_linear_u.gif"))

# Auto-download (Colab only)
try:
    from google.colab import files
    files.download(train_png)
    files.download(diag_png)
    files.download(os.path.join(OUTDIR, "nonforced_snapshots_eta.png"))
    files.download(os.path.join(OUTDIR, "nonforced_snapshots_u.png"))
    files.download(gif_eta)
    files.download(hist_path)
    files.download(eval_path)
except Exception:
    pass