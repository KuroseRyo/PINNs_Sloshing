# ============================================================
# PINN for 1D Shallow Water (SWE) with horizontal forcing (sloshing)
#   rest/flat start: η(x,0)=0, u(x,0)=0 and also η_t(x,0)=0, u_t(x,0)=0
#   + Hard-IC via gate s(t): s(0)=0, s'(0)=0, s(T)=1
#   + BC switch:
#       - BC_MODE="hard": u = sin(pi x/L) * (...)  -> u(0,t)=u(L,t)=0 always
#       - BC_MODE="soft": u = (...) and add BC loss: u(0,t)=u(L,t)=0 (MSE)
#   + Fixed sample counts (same as code2; no scaling-by-domain)
#   + PDE loss: mass + momentum
#   + IC diagnostic loss (sanity; should be ~0)
#   + Mass constraint: mean(η)=0
#   + RAR + ReduceLROnPlateau + optional LBFGS finishing
#   + Forced linear modal benchmark for overlay & error
#   + GIF to /content/Downloads and auto-download on Colab
#
# [ADDED]
#   + FVM (Finite Volume + HLL + SSP-RK2) reference with the same forcing settings
#   + Overlay: PINN vs Linear-forced vs FVM
#   + Relative rel-RMS: PINN vs Linear-forced AND PINN vs FVM
# ============================================================

import math, time, io, os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ===================== USER SWITCH =====================
BC_MODE = "soft"   # "hard" or "soft"
# =======================================================

# ---------------- Device / dtype / seed ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

# ---------------- Physics / domain ----------------
g  = 9.81
Lx = 2.0
h0 = 1.0

# Natural mode (reference only)
n_mode = 1
k1 = n_mode * math.pi / Lx
c  = math.sqrt(g * h0)
omega1 = k1 * c
T1 = 2.0 * math.pi / omega1

# ---------------- Initial condition (STATIC & FLAT) ----------------
phi = 0.0  # unused

# ---------------- Forcing (base excitation) ----------------
X0       = 0.02            # [m]
omega_d  = omega1          # [rad/s]
phase_d  = 0.0
Td       = 2.0 * math.pi / omega_d
T_ramp   = 1.0 * Td

# Train window
N_drive_periods = 2
T = N_drive_periods * Td

# ---------------- PINN hyperparams (FIXED) ----------------
N_f   = 100_000
N_ic  = 5_000
N_bc  = 10_000   # used for BC loss (soft) and diagnostics (hard/soft)
M_t   = 64
M_x   = 256

E_adam  = 10000
E_lbfgs = True

# ---------------- Loss weights ----------------
w_fm, w_fp        = 1.0, 1.0
w_ic_eta, w_ic_u  = 1.0, 1.0     # diagnostic only (hard-ICなので理想~0)
w_bc_u            = 100.0          # used ONLY when BC_MODE="soft"
w_mass            = 5.0

# ---------------- Time stratified sampling ----------------
TIME_BINS = 100

def sample_stratified_time(N: int, Tfinal: float, bins: int, device=device, dtype=None):
    """Stratified sampling on [0,Tfinal], uniform in each bin."""
    if dtype is None:
        dtype = torch.get_default_dtype()
    bins = int(bins)
    N = int(N)
    base = N // bins
    rem  = N - base * bins

    parts = []
    for i in range(bins):
        ni = base + (1 if i < rem else 0)
        if ni <= 0:
            continue
        t0 = (i / bins) * Tfinal
        t1 = ((i + 1) / bins) * Tfinal
        u = torch.rand(ni, 1, device=device, dtype=dtype)
        parts.append(t0 + (t1 - t0) * u)

    t = torch.cat(parts, dim=0)
    idx = torch.randperm(t.shape[0], device=device)
    t = t[idx]
    if t.shape[0] != N:
        t = (torch.rand(N, 1, device=device, dtype=dtype) * Tfinal)
    return t

# ---------------- Utilities ----------------
def d(y, x):
    return grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]

def d_eval(y, x):
    """Derivative for RAR scoring (no create_graph to reduce memory)."""
    return grad(y, x, torch.ones_like(y), create_graph=False, retain_graph=True)[0]

def ramp_cos_torch(t, Tr):
    Trt = torch.as_tensor(Tr, device=t.device, dtype=t.dtype)
    s = torch.clamp(t / (Trt + 1e-12), 0.0, 1.0)
    return 0.5 * (1.0 - torch.cos(math.pi * s))

def ax_torch(t):
    """
    a_x(t) [m/s^2], with ramp so that a_x(0)=0.
    We use momentum residual as: ... + h * a_x(t) = 0
    so define a_x(t) with a minus sign to represent -X0 ω^2 sin(...)
    """
    rt = ramp_cos_torch(t, T_ramp)
    return (-X0 * (omega_d**2) * torch.sin(omega_d * t + phase_d)) * rt

def ramp_cos_np(t, Tr):
    s = np.clip(t / (Tr + 1e-12), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * s))

def ax_np(t):
    rt = ramp_cos_np(t, T_ramp)
    return (-X0 * (omega_d**2) * np.sin(omega_d * t + phase_d)) * rt

# ---------------- Hard-IC gate ----------------
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
    def __init__(self, width=40, depth=8, bc_mode="hard"):
        super().__init__()
        self.bc_mode = bc_mode

        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 2)]
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

        out = self.net(torch.cat([x_nd, t_nd], dim=1))
        eta_hat = out[:, :1]
        u_hat   = out[:, 1:2] * c  # physical scale

        s_eta = s_exp2_norm(t, T, self.a_eta)
        s_u   = s_exp2_norm(t, T, self.a_u)

        # Hard-IC (rest/flat)
        eta = s_eta * eta_hat

        # BC handling
        u_core = s_u * u_hat  # this is the "raw physical" u after hard-IC gate
        if self.bc_mode == "hard":
            b = torch.sin(math.pi * x_nd)  # b(0)=0, b(1)=sin(pi)=0
            u = b * u_core
        else:
            # soft: no envelope; boundary will be enforced by BC loss
            u = u_core

        return eta, u

model = MLP(bc_mode=BC_MODE).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="min", factor=0.5, patience=800, threshold=1e-6, min_lr=1e-6
)

# ---------------- Training points ----------------
x_f = (torch.rand(N_f, 1, device=device) * Lx)
t_f = sample_stratified_time(N_f, T, TIME_BINS, device=device, dtype=torch.get_default_dtype())
x_f = x_f.detach().requires_grad_(True)
t_f = t_f.detach().requires_grad_(True)

# IC points at t=0 (diagnostic)
x_ic = (torch.rand(N_ic, 1, device=device) * Lx)
t_ic = torch.zeros_like(x_ic, device=device)
with torch.no_grad():
    eta0_ic = torch.zeros_like(x_ic)
    u0_ic   = torch.zeros_like(x_ic)

# BC points (used for loss in soft mode, diagnostic otherwise)
t_bcL = (torch.rand(N_bc//2, 1, device=device) * T)
x_bcL = torch.zeros_like(t_bcL, device=device)
t_bcR = (torch.rand(N_bc - N_bc//2, 1, device=device) * T)
x_bcR = torch.full_like(t_bcR, Lx, device=device)

# Mass times
t_mass = torch.linspace(0.0, T, M_t, device=device).reshape(-1,1)

# ---------------- Losses ----------------
def losses():
    eta_f, u_f = model(x_f, t_f)
    h_f = h0 + eta_f
    q_f = h_f * u_f

    r_mass = d(h_f, t_f) + d(q_f, x_f)

    flux = h_f*u_f*u_f + 0.5*g*h_f*h_f
    r_mom  = d(q_f, t_f) + d(flux, x_f) + h_f * ax_torch(t_f)

    L_fm = torch.mean(r_mass**2)
    L_fp = torch.mean(r_mom**2)

    # IC diagnostic
    eta_i, u_i = model(x_ic, t_ic)
    L_ic_eta = torch.mean((eta_i - eta0_ic)**2)
    L_ic_u   = torch.mean((u_i   - u0_ic  )**2)

    # BC loss (ONLY when soft)
    if BC_MODE == "soft":
        _, u_L = model(x_bcL, t_bcL)
        _, u_R = model(x_bcR, t_bcR)
        L_bc_u = 0.5*(torch.mean(u_L**2) + torch.mean(u_R**2))
    else:
        L_bc_u = torch.zeros((), device=device)

    # Mass constraint: mean(η)=0
    L_mass = 0.0
    for tk in t_mass:
        x_s = torch.linspace(0.0, Lx, M_x, device=device).reshape(-1,1)
        t_s = torch.full_like(x_s, float(tk.item()))
        eta_s, _ = model(x_s, t_s)
        L_mass = L_mass + (torch.mean(eta_s))**2
    L_mass = L_mass / len(t_mass)

    total = (
        w_fm*L_fm + w_fp*L_fp +
        w_ic_eta*L_ic_eta + w_ic_u*L_ic_u +
        w_bc_u*L_bc_u +
        w_mass*L_mass
    )

    # diagnostics
    with torch.no_grad():
        h_min = float(torch.min(h_f).cpu())
        _, uL = model(x_bcL, t_bcL)
        _, uR = model(x_bcR, t_bcR)
        u_wall_max = float(torch.max(torch.abs(torch.cat([uL, uR], dim=0))).cpu())

    return total, {
        "fm":L_fm, "fp":L_fp,
        "ic_e":L_ic_eta, "ic_u":L_ic_u,
        "bc_u":L_bc_u,
        "mass":L_mass,
        "hmin": torch.tensor(h_min, device=device),
        "uwall": torch.tensor(u_wall_max, device=device),
    }

# ---------------- RAR ----------------
RAR_Nx = 512
RAR_Nt = 64
RAR_ADD_K = 20000

def _set_param_grad(flag: bool):
    for p in model.parameters():
        p.requires_grad_(flag)

def rar_add_points(add_k=RAR_ADD_K, Nx=RAR_Nx, Nt=RAR_Nt):
    global x_f, t_f

    xg = torch.linspace(0.0, Lx, Nx, device=device).reshape(-1,1)
    tg = torch.linspace(0.0, T,  Nt, device=device).reshape(-1,1)
    X, TT = torch.meshgrid(xg.squeeze(1), tg.squeeze(1), indexing="ij")
    Xc = X.reshape(-1,1).detach().requires_grad_(True)
    Tc = TT.reshape(-1,1).detach().requires_grad_(True)

    was = [p.requires_grad for p in model.parameters()]
    _set_param_grad(False)

    try:
        eta, u = model(Xc, Tc)
        h = h0 + eta
        q = h * u

        r_mass = d_eval(h, Tc) + d_eval(q, Xc)
        flux   = h*u*u + 0.5*g*h*h
        r_mom  = d_eval(q, Tc) + d_eval(flux, Xc) + h * ax_torch(Tc)

        score = (r_mass**2 + r_mom**2).detach().squeeze(1)
    finally:
        for p, fl in zip(model.parameters(), was):
            p.requires_grad_(fl)

    add_k = int(min(add_k, score.numel()))
    vals, idx = torch.topk(score, k=add_k, largest=True, sorted=False)

    x_new = Xc.detach()[idx].reshape(-1,1)
    t_new = Tc.detach()[idx].reshape(-1,1)

    x_f = torch.cat([x_f.detach(), x_new], dim=0).detach().requires_grad_(True)
    t_f = torch.cat([t_f.detach(), t_new], dim=0).detach().requires_grad_(True)

    print(f"[RAR] added {add_k} points | collocation N_f={x_f.shape[0]} | mean(topk)={vals.mean().item():.3e}")

rar_epochs = sorted(list(set([
    int(0.25 * E_adam),
    int(0.50 * E_adam),
    int(0.75 * E_adam),
])))
rar_epochs = [e for e in rar_epochs if e >= 1]
print("[RAR] will trigger at epochs:", rar_epochs, "| BC_MODE=", BC_MODE)

# ---------------- Linear forced benchmark (modal ODE; rest/flat start) ----------------
def linear_forced_modal_solution(times, xgrid, n_modes_odd=9):
    odd_ns = np.array([2*i+1 for i in range(n_modes_odd)], dtype=int)
    omegas = np.sqrt(g*h0) * (odd_ns * np.pi / Lx)
    Fcoef = (4.0*h0/Lx)

    eta_n  = np.zeros((len(odd_ns),), dtype=float)
    eta_dn = np.zeros((len(odd_ns),), dtype=float)

    t_eval = np.array(times, dtype=float)
    t0 = float(t_eval[0])
    t1 = float(t_eval[-1])

    n_steps = max(4000, int(2000 * (t1 - t0) / Td))
    dt = (t1 - t0) / n_steps

    eta_out  = np.zeros((len(t_eval), len(odd_ns)), dtype=float)
    eta_dout = np.zeros((len(t_eval), len(odd_ns)), dtype=float)

    def f_rhs(t, y):
        a = ax_np(t)
        return -(omegas**2) * y + (Fcoef * a) * np.ones_like(y)

    y  = eta_n.copy()
    yd = eta_dn.copy()

    idx = 0
    t = t0
    for step in range(n_steps+1):
        if idx < len(t_eval) and abs(t - t_eval[idx]) <= 0.5*dt + 1e-12:
            eta_out[idx,:]  = y
            eta_dout[idx,:] = yd
            idx += 1
            if idx >= len(t_eval):
                break

        # RK4 for y'' = f_rhs(t,y)  (written as system: y' = yd, yd' = f_rhs)
        k1y = yd
        k1v = f_rhs(t, y)

        k2y = yd + 0.5*dt*k1v
        k2v = f_rhs(t + 0.5*dt, y + 0.5*dt*k1y)

        k3y = yd + 0.5*dt*k2v
        k3v = f_rhs(t + 0.5*dt, y + 0.5*dt*k2y)

        k4y = yd + dt*k3v
        k4v = f_rhs(t + dt, y + dt*k3y)

        y  = y  + (dt/6.0)*(k1y + 2*k2y + 2*k3y + k4y)
        yd = yd + (dt/6.0)*(k1v + 2*k2v + 2*k3v + k4v)
        t += dt

    x = np.array(xgrid, dtype=float).reshape(-1)
    eta_xt = np.zeros((len(t_eval), len(x)), dtype=float)
    u_xt   = np.zeros((len(t_eval), len(x)), dtype=float)

    for j, n in enumerate(odd_ns):
        k = n*np.pi/Lx
        eta_xt += eta_out[:, j:j+1] * np.cos(k * x)[None, :]
        u_amp  = -(1.0/(h0*k)) * eta_dout[:, j:j+1]
        u_xt  += u_amp * np.sin(k * x)[None, :]

    return eta_xt, u_xt

# ====================== 追加（非加振コード同様の保存設定） ======================
OUTDIR = "/content/Downloads"
os.makedirs(OUTDIR, exist_ok=True)

LOG_EVERY   = 50
PRINT_EVERY = 500
EVAL_NT     = 200
EVAL_NX     = 401
SNAP_TIMES  = [0.0, 0.25*T, 0.50*T, 0.75*T, 1.00*T]
# =============================================================================

# ---------------- Train ----------------
history = {
    "epoch": [],
    "lr": [],
    "L_tot": [],
    "L_fm": [], "L_fp": [],
    "L_ic_eta": [], "L_ic_u": [],
    "L_bc_u": [],
    "L_mass": [],
    "hmin": [],
    "uwall": [],
    "Nf": [],
}

t0_train = time.time()
for ep in range(1, E_adam+1):
    if ep in rar_epochs:
        rar_add_points(add_k=RAR_ADD_K, Nx=RAR_Nx, Nt=RAR_Nt)

    opt.zero_grad(set_to_none=True)
    Ltot, dct = losses()
    Ltot.backward()
    opt.step()

    scheduler.step(Ltot.item())
    lr_now = opt.param_groups[0]["lr"]

    if ep % LOG_EVERY == 0 or ep == 1:
        history["epoch"].append(ep)
        history["lr"].append(float(lr_now))
        history["L_tot"].append(float(Ltot.item()))
        history["L_fm"].append(float(dct["fm"].item()))
        history["L_fp"].append(float(dct["fp"].item()))
        history["L_ic_eta"].append(float(dct["ic_e"].item()))
        history["L_ic_u"].append(float(dct["ic_u"].item()))
        history["L_bc_u"].append(float(dct["bc_u"].item()))
        history["L_mass"].append(float(dct["mass"].item()))
        history["hmin"].append(float(dct["hmin"].item()))
        history["uwall"].append(float(dct["uwall"].item()))
        history["Nf"].append(int(x_f.shape[0]))

    if ep % PRINT_EVERY == 0 or ep == 1:
        print(
            f"[Adam {ep:5d}] tot={Ltot.item():.3e} "
            f"fm={dct['fm'].item():.2e} fp={dct['fp'].item():.2e} "
            f"IC(η/u)=({dct['ic_e'].item():.2e}/{dct['ic_u'].item():.2e}) "
            f"BC_u={dct['bc_u'].item():.2e} "
            f"mass={dct['mass'].item():.2e} "
            f"hmin≈{dct['hmin'].item():+.3e} "
            f"|u|wall_max≈{dct['uwall'].item():.3e} "
            f"lr={lr_now:.2e} Nf={x_f.shape[0]}"
        )

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

print(f"Done in {time.time()-t0_train:.1f}s")

# ====================== [ADDED] FVM reference (same forcing settings) ======================
# Baseline FVM settings (your chosen reference)
FVM_NX    = 800
FVM_CFL   = 0.30
FVM_RECON = True
FVM_H_MIN = 1e-8

def fvm_minmod(a, b):
    out = np.zeros_like(a)
    mask = (a*b) > 0.0
    out[mask] = np.where(np.abs(a[mask]) < np.abs(b[mask]), a[mask], b[mask])
    return out

def fvm_extend_ghosts(h, u):
    NX = h.size
    h_ext = np.zeros(NX + 2)
    u_ext = np.zeros(NX + 2)
    h_ext[1:-1] = h
    u_ext[1:-1] = u
    h_ext[0]  = h[0]
    u_ext[0]  = -u[0]
    h_ext[-1] = h[-1]
    u_ext[-1] = -u[-1]
    return h_ext, u_ext

def fvm_hll_flux(hL, uL, hR, uR):
    mL = hL*uL
    mR = hR*uR
    cL = np.sqrt(g*np.maximum(hL, 0.0))
    cR = np.sqrt(g*np.maximum(hR, 0.0))
    sL = np.minimum(uL - cL, uR - cR)
    sR = np.maximum(uL + cL, uR + cR)

    FL1 = mL
    FL2 = mL*uL + 0.5*g*hL*hL
    FR1 = mR
    FR2 = mR*uR + 0.5*g*hR*hR

    denom = (sR - sL + 1e-14)
    F1 = np.where(sL >= 0, FL1,
         np.where(sR <= 0, FR1,
                  (sR*FL1 - sL*FR1 + sL*sR*(hR - hL)) / denom))
    F2 = np.where(sL >= 0, FL2,
         np.where(sR <= 0, FR2,
                  (sR*FL2 - sL*FR2 + sL*sR*(mR - mL)) / denom))
    return F1, F2

def fvm_reconstruct_interfaces(h_ext, u_ext):
    dh = np.zeros_like(h_ext)
    du = np.zeros_like(u_ext)
    dh[1:-1] = fvm_minmod(h_ext[2:] - h_ext[1:-1], h_ext[1:-1] - h_ext[:-2])
    du[1:-1] = fvm_minmod(u_ext[2:] - u_ext[1:-1], u_ext[1:-1] - u_ext[:-2])

    hL = h_ext[:-1] + 0.5*dh[:-1]
    uL = u_ext[:-1] + 0.5*du[:-1]
    hR = h_ext[1:]  - 0.5*dh[1:]
    uR = u_ext[1:]  - 0.5*du[1:]

    hL = np.maximum(hL, FVM_H_MIN)
    hR = np.maximum(hR, FVM_H_MIN)
    return hL, uL, hR, uR

def fvm_compute_dt(h, m, dx, CFL):
    u = np.where(h > 1e-12, m / h, 0.0)
    cwv = np.sqrt(g * np.maximum(h, 0.0))
    smax = np.max(np.abs(u) + cwv)
    return CFL * dx / (smax + 1e-14)

def fvm_rhs(h, m, t, dx, RECON):
    u = np.where(h > 1e-12, m / h, 0.0)
    h_ext, u_ext = fvm_extend_ghosts(h, u)

    if RECON:
        hL, uL, hR, uR = fvm_reconstruct_interfaces(h_ext, u_ext)
    else:
        hL = np.maximum(h_ext[:-1], FVM_H_MIN)
        uL = u_ext[:-1]
        hR = np.maximum(h_ext[1:],  FVM_H_MIN)
        uR = u_ext[1:]

    F1, F2 = fvm_hll_flux(hL, uL, hR, uR)

    dF1 = (F1[1:] - F1[:-1]) / dx
    dF2 = (F2[1:] - F2[:-1]) / dx

    ax = ax_np(float(t))
    S1 = 0.0
    S2 = -h * ax

    dhdt = -dF1 + S1
    dmdt = -dF2 + S2
    return dhdt, dmdt

def fvm_run_snapshots(ts_targets, NX=FVM_NX, CFL=FVM_CFL, RECON=FVM_RECON, enforce_wall_momentum=True):
    """
    Run FVM once and store snapshots exactly at ts_targets (ascending, includes t=0 ok).
    Returns:
      x_cc (cell centers, size NX),
      eta_snap [len(ts), NX], u_snap [len(ts), NX]
    """
    ts_targets = np.array(ts_targets, dtype=float).reshape(-1)
    ts_targets = np.clip(ts_targets, 0.0, T)
    # ensure ascending
    order = np.argsort(ts_targets)
    ts_sorted = ts_targets[order]

    dx = Lx / NX
    x_cc = np.linspace(0.0 + 0.5*dx, Lx - 0.5*dx, NX)

    h = np.full(NX, h0, dtype=float)
    m = np.zeros(NX, dtype=float)

    eta_out = np.zeros((len(ts_sorted), NX), dtype=float)
    u_out   = np.zeros((len(ts_sorted), NX), dtype=float)

    def write(k):
        u = np.where(h > 1e-12, m/h, 0.0)
        eta_out[k,:] = h - h0
        u_out[k,:]   = u

    t = 0.0
    k = 0

    # write any targets at t=0
    while k < len(ts_sorted) and (t >= ts_sorted[k] - 1e-12):
        write(k)
        k += 1

    while t < T - 1e-14 and k < len(ts_sorted):
        dt = fvm_compute_dt(h, m, dx, CFL)
        # do not step beyond next target too far; clamp to hit it (simple)
        dt = min(dt, ts_sorted[k] - t)
        if dt <= 0.0:
            # already at/over target
            write(k)
            k += 1
            continue

        # SSP-RK2
        dh1, dm1 = fvm_rhs(h, m, t, dx, RECON)
        h1 = h + dt*dh1
        m1 = m + dt*dm1
        h1 = np.maximum(h1, FVM_H_MIN)

        dh2, dm2 = fvm_rhs(h1, m1, t + dt, dx, RECON)
        h2 = 0.5*(h + h1 + dt*dh2)
        m2 = 0.5*(m + m1 + dt*dm2)
        h2 = np.maximum(h2, FVM_H_MIN)

        if enforce_wall_momentum:
            m2[0]  = 0.0
            m2[-1] = 0.0

        h, m = h2, m2
        t += dt

        while k < len(ts_sorted) and (t >= ts_sorted[k] - 1e-12):
            write(k)
            k += 1

    # reorder back to original ts order
    eta_back = np.zeros_like(eta_out)
    u_back   = np.zeros_like(u_out)
    eta_back[order,:] = eta_out
    u_back[order,:]   = u_out
    return x_cc, eta_back, u_back

def fvm_interp_to_x(x_src, f_src, x_dst):
    # np.interp requires x_dst within range; we hold boundary constant beyond edges
    return np.interp(x_dst, x_src, f_src, left=f_src[0], right=f_src[-1])
# =============================================================================

# ---------------- Benchmark helpers ----------------
@torch.no_grad()
def make_linear_benchmark(nx=400, n_frames=120, n_modes_odd=9):
    xs = np.linspace(0.0, Lx, nx)
    ts = np.linspace(0.0, T, n_frames)
    eta_lin, u_lin = linear_forced_modal_solution(ts, xs, n_modes_odd=n_modes_odd)
    return xs, ts, eta_lin, u_lin

@torch.no_grad()
def eval_forced_linear_error(nx=256, nt=80, n_modes_odd=9):
    xs = np.linspace(0.0, Lx, nx)
    ts = np.linspace(0.0, T, nt)

    eta_lin, u_lin = linear_forced_modal_solution(ts, xs, n_modes_odd=n_modes_odd)

    xs_t = torch.tensor(xs, device=device, dtype=torch.get_default_dtype()).reshape(-1,1)
    errs_eta, errs_u = [], []
    A_eta = max(1e-6, float(np.max(np.abs(eta_lin))) )
    A_u   = max(1e-6, float(np.max(np.abs(u_lin)))   )

    for j, tj in enumerate(ts):
        t_t = torch.full_like(xs_t, float(tj))
        eta_p, u_p = model(xs_t, t_t)
        eta_p = eta_p.squeeze().cpu().numpy()
        u_p   = u_p.squeeze().cpu().numpy()

        e_eta = np.sqrt(np.mean((eta_p - eta_lin[j,:])**2)) / A_eta
        e_u   = np.sqrt(np.mean((u_p   - u_lin[j,:])**2)) / A_u
        errs_eta.append(float(e_eta))
        errs_u.append(float(e_u))

    return float(np.mean(errs_eta)), float(np.mean(errs_u)), float(np.max(errs_eta)), float(np.max(errs_u))

# [existing] PINN vs linear-forced
e_eta_mean, e_u_mean, e_eta_max, e_u_max = eval_forced_linear_error()
print(f"[Forced-linear agreement | BC_MODE={BC_MODE}] mean rel-RMS: eta={e_eta_mean:.3e}, u={e_u_mean:.3e} | "
      f"max: eta={e_eta_max:.3e}, u={e_u_max:.3e}")

# [ADDED] PINN vs FVM (same evaluation grid/time count as linear error)
@torch.no_grad()
def eval_forced_fvm_error(nx=256, nt=80):
    xs = np.linspace(0.0, Lx, nx)
    ts = np.linspace(0.0, T, nt)

    # FVM snapshots at ts, then interpolate to xs for each frame
    xcc, eta_cc, u_cc = fvm_run_snapshots(ts, NX=FVM_NX, CFL=FVM_CFL, RECON=FVM_RECON, enforce_wall_momentum=True)
    eta_fvm = np.zeros((nt, nx), dtype=float)
    u_fvm   = np.zeros((nt, nx), dtype=float)
    for j in range(nt):
        eta_fvm[j,:] = fvm_interp_to_x(xcc, eta_cc[j,:], xs)
        u_fvm[j,:]   = fvm_interp_to_x(xcc, u_cc[j,:],   xs)

    xs_t = torch.tensor(xs, device=device, dtype=torch.get_default_dtype()).reshape(-1,1)
    errs_eta, errs_u = [], []

    A_eta = max(1e-6, float(np.max(np.abs(eta_fvm))))
    A_u   = max(1e-6, float(np.max(np.abs(u_fvm))))

    for j, tj in enumerate(ts):
        t_t = torch.full_like(xs_t, float(tj))
        eta_p, u_p = model(xs_t, t_t)
        eta_p = eta_p.squeeze().cpu().numpy()
        u_p   = u_p.squeeze().cpu().numpy()

        e_eta = np.sqrt(np.mean((eta_p - eta_fvm[j,:])**2)) / A_eta
        e_u   = np.sqrt(np.mean((u_p   - u_fvm[j,:])**2)) / A_u
        errs_eta.append(float(e_eta))
        errs_u.append(float(e_u))

    return float(np.mean(errs_eta)), float(np.mean(errs_u)), float(np.max(errs_eta)), float(np.max(errs_u))

e_eta_mean_fvm, e_u_mean_fvm, e_eta_max_fvm, e_u_max_fvm = eval_forced_fvm_error()
print(f"[Forced-FVM agreement   | BC_MODE={BC_MODE}] mean rel-RMS: eta={e_eta_mean_fvm:.3e}, u={e_u_mean_fvm:.3e} | "
      f"max: eta={e_eta_max_fvm:.3e}, u={e_u_max_fvm:.3e}")

# ---------------- Visualization GIF (PINN vs forced-linear (+FVM)) ----------------
@torch.no_grad()
def make_gif_eta_overlay_forced(out=None, n_frames=120, nx_plot=400, n_modes_odd=9, margin=0.25):
    if out is None:
        out = f"/content/Downloads/pinn_{BC_MODE}_BC_vs_linear_forced_eta.gif"
    try:
        os.makedirs(os.path.dirname(out), exist_ok=True)
    except:
        pass

    xs, ts, eta_lin, _ = make_linear_benchmark(nx=nx_plot, n_frames=n_frames, n_modes_odd=n_modes_odd)
    xs_t = torch.tensor(xs, device=device, dtype=torch.get_default_dtype()).reshape(-1,1)

    # [ADDED] FVM on same ts, interpolate to xs
    xcc, eta_cc, _u_cc = fvm_run_snapshots(ts, NX=FVM_NX, CFL=FVM_CFL, RECON=FVM_RECON, enforce_wall_momentum=True)
    eta_fvm = np.zeros_like(eta_lin)
    for j in range(len(ts)):
        eta_fvm[j,:] = fvm_interp_to_x(xcc, eta_cc[j,:], xs)

    eta_amp = 0.0
    for tj in ts:
        t_t = torch.full_like(xs_t, float(tj))
        eta_p, _ = model(xs_t, t_t)
        eta_amp = max(eta_amp, float(torch.max(torch.abs(eta_p)).cpu()))
    eta_amp = max(eta_amp, float(np.max(np.abs(eta_lin))))
    eta_amp = max(eta_amp, float(np.max(np.abs(eta_fvm))))
    y_lim = (1.0 + margin) * max(eta_amp, 1e-6)

    frames = []
    for j, tj in enumerate(ts):
        t_t = torch.full_like(xs_t, float(tj))
        eta_p, _ = model(xs_t, t_t)
        etaP = eta_p.squeeze().cpu().numpy()
        etaL = eta_lin[j,:]
        etaG = eta_fvm[j,:]

        mse_x_lin = float(np.mean((etaP - etaL)**2))
        mse_x_fvm = float(np.mean((etaP - etaG)**2))
        a_now = ax_np(float(tj))

        fig = plt.figure(figsize=(6.9,3.6))
        plt.plot(xs, etaP, lw=2.2, label=f"PINN η ({BC_MODE}-BC)")
        plt.plot(xs, etaL, lw=2.2, ls="--", label="Linear-forced η (modal)")
        plt.plot(xs, etaG, lw=2.2, ls=":",  label=f"FVM η (NX={FVM_NX},CFL={FVM_CFL},RECON={FVM_RECON})")
        plt.axhline(0.0, ls=":", lw=1)
        plt.ylim(-y_lim, y_lim); plt.xlim(0.0, Lx)
        plt.xlabel("x [m]"); plt.ylabel("η(x,t) [m]")
        plt.title(
            f"Forced sloshing  t={tj:.2f}s  a_x={a_now:+.3f} m/s²   "
            f"MSE_x(P-L)≈{mse_x_lin:.2e}  MSE_x(P-FVM)≈{mse_x_fvm:.2e}"
        )
        plt.grid(True, alpha=0.3); plt.legend(loc="upper right", fontsize=9)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=170)
        plt.close(fig)
        buf.seek(0)
        import PIL.Image as Image
        frames.append(np.array(Image.open(buf).convert("RGB")))
        buf.close()

    imageio.mimsave(out, frames, duration=max(T/n_frames, 0.03))
    print(f"Saved GIF to: {out}")
    try:
        from google.colab import files
        files.download(out)
    except Exception:
        pass

make_gif_eta_overlay_forced()

# ====================== 追加（非加振コード同様の保存出力） ======================
# ---- Save training history (.npz) ----
hist_path = os.path.join(OUTDIR, f"forced_pinn_{BC_MODE}_training_history.npz")
np.savez(hist_path, **{k: np.array(v) for k, v in history.items()})
print("Saved:", hist_path)

# ---- Plot training curves ----
ep_arr = np.array(history["epoch"], dtype=int)
fig = plt.figure(figsize=(10.2, 6.2))
plt.semilogy(ep_arr, history["L_tot"], label="L_total")
plt.semilogy(ep_arr, history["L_fm"],  label="L_mass_PDE")
plt.semilogy(ep_arr, history["L_fp"],  label="L_mom_PDE")
plt.semilogy(ep_arr, history["L_bc_u"], label="L_bc_u (soft only)")
plt.semilogy(ep_arr, history["L_mass"], label="L_mean_eta")
plt.semilogy(ep_arr, np.array(history["L_ic_eta"]) + np.array(history["L_ic_u"]), label="L_IC_diag (eta+u)")
plt.xlabel("epoch"); plt.ylabel("loss (log scale)")
plt.title(f"Training curves (forced SWE PINN)  BC_MODE={BC_MODE}")
plt.grid(True, alpha=0.3); plt.legend()
train_png = os.path.join(OUTDIR, f"forced_{BC_MODE}_training_curves.png")
plt.tight_layout(); plt.savefig(train_png, dpi=180); plt.close(fig)
print("Saved:", train_png)

# ---- Evaluation time-series vs forced-linear (+FVM) + wall/mass diagnostics ----
@torch.no_grad()
def eval_forced_time_series(nx=EVAL_NX, nt=EVAL_NT, n_modes_odd=9):
    xs_np = np.linspace(0.0, Lx, nx)
    ts_np = np.linspace(0.0, T, nt)

    eta_lin, u_lin = linear_forced_modal_solution(ts_np, xs_np, n_modes_odd=n_modes_odd)
    A_eta_lin = max(1e-6, float(np.max(np.abs(eta_lin))))
    A_u_lin   = max(1e-6, float(np.max(np.abs(u_lin))))

    # [ADDED] FVM on same (ts,xs)
    xcc, eta_cc, u_cc = fvm_run_snapshots(ts_np, NX=FVM_NX, CFL=FVM_CFL, RECON=FVM_RECON, enforce_wall_momentum=True)
    eta_fvm = np.zeros((nt, nx), dtype=float)
    u_fvm   = np.zeros((nt, nx), dtype=float)
    for i in range(nt):
        eta_fvm[i,:] = fvm_interp_to_x(xcc, eta_cc[i,:], xs_np)
        u_fvm[i,:]   = fvm_interp_to_x(xcc, u_cc[i,:],   xs_np)

    A_eta_fvm = max(1e-6, float(np.max(np.abs(eta_fvm))))
    A_u_fvm   = max(1e-6, float(np.max(np.abs(u_fvm))))

    xs_t = torch.tensor(xs_np, device=device, dtype=torch.get_default_dtype()).reshape(-1,1)
    x0 = torch.zeros((1,1), device=device, dtype=torch.get_default_dtype())
    xL = torch.full((1,1), Lx, device=device, dtype=torch.get_default_dtype())

    e_eta = np.zeros(nt, dtype=float)
    e_u   = np.zeros(nt, dtype=float)
    e_eta_fvm_ts = np.zeros(nt, dtype=float)
    e_u_fvm_ts   = np.zeros(nt, dtype=float)

    mean_eta = np.zeros(nt, dtype=float)
    uwL = np.zeros(nt, dtype=float)
    uwR = np.zeros(nt, dtype=float)

    for i, tj in enumerate(ts_np):
        tcol = torch.full_like(xs_t, float(tj))
        eta_p, u_p = model(xs_t, tcol)
        eta_p_np = eta_p.squeeze().cpu().numpy()
        u_p_np   = u_p.squeeze().cpu().numpy()

        # PINN vs linear-forced
        e_eta[i] = float(np.sqrt(np.mean((eta_p_np - eta_lin[i,:])**2)) / A_eta_lin)
        e_u[i]   = float(np.sqrt(np.mean((u_p_np   - u_lin[i,:])**2)) / A_u_lin)

        # [ADDED] PINN vs FVM
        e_eta_fvm_ts[i] = float(np.sqrt(np.mean((eta_p_np - eta_fvm[i,:])**2)) / A_eta_fvm)
        e_u_fvm_ts[i]   = float(np.sqrt(np.mean((u_p_np   - u_fvm[i,:])**2)) / A_u_fvm)

        mean_eta[i] = float(np.mean(eta_p_np))

        t1 = torch.full((1,1), float(tj), device=device, dtype=torch.get_default_dtype())
        _, u0 = model(x0, t1)
        _, u1 = model(xL, t1)
        uwL[i] = float(torch.abs(u0).cpu())
        uwR[i] = float(torch.abs(u1).cpu())

    return ts_np, e_eta, e_u, e_eta_fvm_ts, e_u_fvm_ts, mean_eta, uwL, uwR

t_arr, e_eta_t, e_u_t, e_eta_fvm_t, e_u_fvm_t, mean_eta_t, uwL_t, uwR_t = eval_forced_time_series()

print(f"[Forced vs linear | BC_MODE={BC_MODE}] mean rel-RMS: eta={e_eta_t.mean():.3e}, u={e_u_t.mean():.3e}")
print(f"[Forced vs linear | BC_MODE={BC_MODE}] max  rel-RMS: eta={e_eta_t.max():.3e}, u={e_u_t.max():.3e}")
print(f"[Forced vs FVM    | BC_MODE={BC_MODE}] mean rel-RMS: eta={e_eta_fvm_t.mean():.3e}, u={e_u_fvm_t.mean():.3e}")
print(f"[Forced vs FVM    | BC_MODE={BC_MODE}] max  rel-RMS: eta={e_eta_fvm_t.max():.3e}, u={e_u_fvm_t.max():.3e}")
print(f"[Wall violation | BC_MODE={BC_MODE}] max |u(0,t)|={uwL_t.max():.3e}, max |u(L,t)|={uwR_t.max():.3e}")
print(f"[Mass condition | BC_MODE={BC_MODE}] max |mean_eta(t)|={np.max(np.abs(mean_eta_t)):.3e}")

eval_path = os.path.join(OUTDIR, f"forced_{BC_MODE}_eval_timeseries.npz")
np.savez(
    eval_path,
    t=t_arr,
    e_eta=e_eta_t, e_u=e_u_t,
    e_eta_fvm=e_eta_fvm_t, e_u_fvm=e_u_fvm_t,   # [ADDED]
    mean_eta=mean_eta_t, uwall_L=uwL_t, uwall_R=uwR_t
)
print("Saved:", eval_path)

# ---- Plot diagnostics time-series ----
fig = plt.figure(figsize=(10.5, 7.6))
ax1 = plt.subplot(3,1,1)
ax1.plot(t_arr, e_eta_t,      label="rel-RMS error (eta) vs Linear")
ax1.plot(t_arr, e_u_t,        label="rel-RMS error (u)   vs Linear")
ax1.plot(t_arr, e_eta_fvm_t,  label="rel-RMS error (eta) vs FVM")
ax1.plot(t_arr, e_u_fvm_t,    label="rel-RMS error (u)   vs FVM")
ax1.set_ylabel("relative RMS")
ax1.grid(True, alpha=0.3); ax1.legend(fontsize=9)

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

diag_png = os.path.join(OUTDIR, f"forced_{BC_MODE}_diagnostics_timeseries.png")
plt.tight_layout(); plt.savefig(diag_png, dpi=180); plt.close(fig)
print("Saved:", diag_png)

# ---- Snapshot profiles (PINN vs forced-linear (+FVM)) ----
@torch.no_grad()
def plot_forced_snapshots(times, nx=801, n_modes_odd=9):
    xs = np.linspace(0.0, Lx, nx)
    xs_t = torch.tensor(xs, device=device, dtype=torch.get_default_dtype()).reshape(-1,1)

    times = [float(ti) for ti in times]
    eta_lin, u_lin = linear_forced_modal_solution(times, xs, n_modes_odd=n_modes_odd)

    # [ADDED] FVM snapshots at these times, interpolate to xs
    xcc, eta_cc, u_cc = fvm_run_snapshots(times, NX=FVM_NX, CFL=FVM_CFL, RECON=FVM_RECON, enforce_wall_momentum=True)
    eta_fvm = np.zeros_like(eta_lin)
    u_fvm   = np.zeros_like(u_lin)
    for j in range(len(times)):
        eta_fvm[j,:] = fvm_interp_to_x(xcc, eta_cc[j,:], xs)
        u_fvm[j,:]   = fvm_interp_to_x(xcc, u_cc[j,:],   xs)

    # eta snapshots
    fig1 = plt.figure(figsize=(10.6, 4.8))
    for j, tcur in enumerate(times):
        tt = torch.full_like(xs_t, tcur)
        eta_p, _ = model(xs_t, tt)
        plt.plot(xs, eta_p.squeeze().cpu().numpy(), lw=2, label=f"PINN t={tcur:.2f}")
        plt.plot(xs, eta_lin[j,:], lw=2, ls="--", label=f"Linear t={tcur:.2f}")
        plt.plot(xs, eta_fvm[j,:], lw=2, ls=":",  label=f"FVM t={tcur:.2f}")
    plt.axhline(0.0, ls=":", lw=1)
    plt.xlabel("x [m]"); plt.ylabel("η(x,t) [m]")
    plt.title(f"Snapshots: η(x,t) PINN vs Linear vs FVM (BC_MODE={BC_MODE})")
    plt.grid(True, alpha=0.3); plt.legend(ncol=3, fontsize=9)
    eta_png = os.path.join(OUTDIR, f"forced_{BC_MODE}_snapshots_eta.png")
    plt.tight_layout(); plt.savefig(eta_png, dpi=180); plt.close(fig1)
    print("Saved:", eta_png)

    # u snapshots
    fig2 = plt.figure(figsize=(10.6, 4.8))
    for j, tcur in enumerate(times):
        tt = torch.full_like(xs_t, tcur)
        _, u_p = model(xs_t, tt)
        plt.plot(xs, u_p.squeeze().cpu().numpy(), lw=2, label=f"PINN t={tcur:.2f}")
        plt.plot(xs, u_lin[j,:], lw=2, ls="--", label=f"Linear t={tcur:.2f}")
        plt.plot(xs, u_fvm[j,:], lw=2, ls=":",  label=f"FVM t={tcur:.2f}")
    plt.axhline(0.0, ls=":", lw=1)
    plt.xlabel("x [m]"); plt.ylabel("u(x,t) [m/s]")
    plt.title(f"Snapshots: u(x,t) PINN vs Linear vs FVM (BC_MODE={BC_MODE})")
    plt.grid(True, alpha=0.3); plt.legend(ncol=3, fontsize=9)
    u_png = os.path.join(OUTDIR, f"forced_{BC_MODE}_snapshots_u.png")
    plt.tight_layout(); plt.savefig(u_png, dpi=180); plt.close(fig2)
    print("Saved:", u_png)

plot_forced_snapshots(SNAP_TIMES)

# ====================== [ADDED REQUEST] Pairwise overlays (1 row x 3 cols) ======================
@torch.no_grad()
def plot_forced_pairwise_triptychs(times, nx=801, n_modes_odd=9):
    """
    For each time in `times`, output a 1x3 figure:
      (1) PINN vs Linear, (2) PINN vs FVM, (3) Linear vs FVM
    for both eta and u. Existing outputs are kept as-is.
    """
    xs = np.linspace(0.0, Lx, nx)
    xs_t = torch.tensor(xs, device=device, dtype=torch.get_default_dtype()).reshape(-1,1)

    times = [float(ti) for ti in times]
    eta_lin, u_lin = linear_forced_modal_solution(times, xs, n_modes_odd=n_modes_odd)

    # FVM snapshots at these times, interpolate to xs
    xcc, eta_cc, u_cc = fvm_run_snapshots(times, NX=FVM_NX, CFL=FVM_CFL, RECON=FVM_RECON, enforce_wall_momentum=True)
    eta_fvm = np.zeros_like(eta_lin)
    u_fvm   = np.zeros_like(u_lin)
    for j in range(len(times)):
        eta_fvm[j,:] = fvm_interp_to_x(xcc, eta_cc[j,:], xs)
        u_fvm[j,:]   = fvm_interp_to_x(xcc, u_cc[j,:],   xs)

    pairwise_eta_paths = []
    pairwise_u_paths   = []

    for j, tcur in enumerate(times):
        tt = torch.full_like(xs_t, tcur)
        eta_p, u_p = model(xs_t, tt)
        eta_p = eta_p.squeeze().cpu().numpy()
        u_p   = u_p.squeeze().cpu().numpy()

        # --- eta: 1x3 ---
        fig = plt.figure(figsize=(15.6, 4.2))
        ax1 = plt.subplot(1,3,1)
        ax1.plot(xs, eta_p, lw=2)
        ax1.plot(xs, eta_lin[j,:], lw=2, ls="--")
        ax1.set_title("PINN vs Linear (η)")
        ax1.set_xlabel("x [m]"); ax1.set_ylabel("η [m]")
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(1,3,2)
        ax2.plot(xs, eta_p, lw=2)
        ax2.plot(xs, eta_fvm[j,:], lw=2, ls=":")
        ax2.set_title("PINN vs FVM (η)")
        ax2.set_xlabel("x [m]")
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(1,3,3)
        ax3.plot(xs, eta_lin[j,:], lw=2, ls="--")
        ax3.plot(xs, eta_fvm[j,:], lw=2, ls=":")
        ax3.set_title("Linear vs FVM (η)")
        ax3.set_xlabel("x [m]")
        ax3.grid(True, alpha=0.3)

        fig.suptitle(f"Pairwise overlays at t={tcur:.3f} s (BC_MODE={BC_MODE})", y=1.02)
        plt.tight_layout()
        eta_trip_png = os.path.join(OUTDIR, f"forced_{BC_MODE}_pairwise_eta_t{j:02d}.png")
        plt.savefig(eta_trip_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", eta_trip_png)
        pairwise_eta_paths.append(eta_trip_png)

        # --- u: 1x3 ---
        fig = plt.figure(figsize=(15.6, 4.2))
        ax1 = plt.subplot(1,3,1)
        ax1.plot(xs, u_p, lw=2)
        ax1.plot(xs, u_lin[j,:], lw=2, ls="--")
        ax1.set_title("PINN vs Linear (u)")
        ax1.set_xlabel("x [m]"); ax1.set_ylabel("u [m/s]")
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(1,3,2)
        ax2.plot(xs, u_p, lw=2)
        ax2.plot(xs, u_fvm[j,:], lw=2, ls=":")
        ax2.set_title("PINN vs FVM (u)")
        ax2.set_xlabel("x [m]")
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(1,3,3)
        ax3.plot(xs, u_lin[j,:], lw=2, ls="--")
        ax3.plot(xs, u_fvm[j,:], lw=2, ls=":")
        ax3.set_title("Linear vs FVM (u)")
        ax3.set_xlabel("x [m]")
        ax3.grid(True, alpha=0.3)

        fig.suptitle(f"Pairwise overlays at t={tcur:.3f} s (BC_MODE={BC_MODE})", y=1.02)
        plt.tight_layout()
        u_trip_png = os.path.join(OUTDIR, f"forced_{BC_MODE}_pairwise_u_t{j:02d}.png")
        plt.savefig(u_trip_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", u_trip_png)
        pairwise_u_paths.append(u_trip_png)

    return pairwise_eta_paths, pairwise_u_paths

pair_eta_paths, pair_u_paths = plot_forced_pairwise_triptychs(SNAP_TIMES)
# =============================================================================

# ---- Auto-download (Colab only) ----
try:
    from google.colab import files
    files.download(hist_path)
    files.download(train_png)
    files.download(eval_path)
    files.download(diag_png)
    files.download(os.path.join(OUTDIR, f"forced_{BC_MODE}_snapshots_eta.png"))
    files.download(os.path.join(OUTDIR, f"forced_{BC_MODE}_snapshots_u.png"))
    # [ADDED REQUEST] download pairwise triptychs
    for p in pair_eta_paths:
        files.download(p)
    for p in pair_u_paths:
        files.download(p)
except Exception:
    pass
# =============================================================================