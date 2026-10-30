"""
MT-DRONE: steady-state (SS) model
=================================
Deep-learning estimation of quantitative magnetization transfer (qMT) parameters
R1a, M0b, R and T2b from steady-state Z-spectra (5 B1 amplitudes x 52 offsets) plus a
24-point selective inversion-recovery (SelIR) curve, acquired at 6.5 mT.

Running `python mt_drone.py` does the whole pipeline:
  1. generate_dataset()       simulate 50,000 x 25 = 1.25M noisy training signals
                              (Morrison-Henkelman steady-state Z-spectrum + SelIR)
  2. train_model()            train a 285 -> 300 -> 300 -> 4 fully connected network and
                              print the mean % error on the 10% synthetic test set
  3. evaluate_experimental()  compare predictions on the phantom data to the fitted
                              ground truth (skipped if ./experimental_data is missing)
  4. log_run()                save plots and a summary to result_history/runN/

Note: Code was written in a hybrid way between me and LLMs. I came up with the structure
and plan, and LLMs helped write the functions. I reviewed the code afterward and ran tests
to check accuracy.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
from pathlib import Path
import time
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS: the values below are the ones used in the manuscript.
# To adapt to a new acquisition protocol, change the offsets / B1 / timing here.
# =============================================================================

# MT parameter ranges (R1a, M0b, R, T2b) — adjust here to propagate everywhere
PARAM_RANGES = {
    'R1a': (0.2,   1.0),
    'M0b': (0.001, 0.2),
    'R':   (12,  70.0),
    'T2b': (8e-6, 25e-6),
}

# Nuisance parameter ranges — adjust here to propagate everywhere
NUISANCE_RANGES = {
    'R1b':         (1,  1),
    'T2a':         (0.030,  2.500),
    'Td':          (0.005, 0.1),
    'Ma0':         (-1.0,  -0.70),  # Ceiling-correlated: high-M0b samples → less complete inversion
    'B0_jitter_hz': (-30,30),  # B0 drift during scan (Hz)
    'B1_scale':     (0.95, 1.05),  # B1 inhomogeneity (±5% RF power variation)
}

# Hyperparameters — adjust here to propagate everywhere
HIDDEN_DIM        = 300    # nodes per hidden layer
N_LAYERS          = 2      # number of hidden layers
LEARNING_RATE     = 0.002
EPOCHS            = 200
WARMUP_EPOCHS     = 5
TRAIN_BATCH_SIZE  = 2048  # batch size during training
GEN_BATCH_SIZE    = 4096   # batch size during data generation
N_SAMPLES         = 50000   # number of base parameter sets
N_AUGMENT         = 25    # nuisance parameter augmentations per base sample
NOISE_STD         = 0.01   # Gaussian noise std added to simulated signals
CONS_LAMBDA       = 0.0    # weight for forward model consistency loss (0 = disabled)
CONS_WARMUP       = 20     # epochs to ramp up consistency loss from 0 → full

# Saturation offset frequencies (Hz), 52 offsets
OFFSETS_HZ = np.array([-36000, -34000, -32000, -30000, -28000, -26000, -24000, -22000,
    -20000, -18000, -16000, -14000, -12000, -10000, -8000, -6000, -5000, -4000,
    -3500, -3000, -2500, -2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000,
    2500, 3000, 3500, 4000, 5000, 6000, 8000, 10000, 12000, 14000, 16000, 18000,
    20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000, 70000], dtype=np.float64)

# Saturation amplitudes B1 = 62.5, 125, 187.5, 250, 500 Hz, in rad/s
W1_RAD = np.array([392.699, 785.398, 1178.097, 1570.796, 3141.593], dtype=np.float64)

# SelIR inversion times (s), 24 points
TI_TIMES = np.array([0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02,
    0.025, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0,
    5.0, 10.0], dtype=np.float64)

# Possible last TI values across experimental samples (MTF50+: 10s, MTF10-40/TRES: 13s).
# The last TI is appended as a normalized extra input feature so the network generalizes.
TI_LAST_OPTIONS = np.array([10.0, 13.0], dtype=np.float64)
TI_LAST_MIN, TI_LAST_MAX = TI_LAST_OPTIONS[0], TI_LAST_OPTIONS[-1]

# Normalization bounds derived from PARAM_RANGES — do not edit directly
PARAM_LO = np.array([PARAM_RANGES['R1a'][0], PARAM_RANGES['M0b'][0], PARAM_RANGES['R'][0], PARAM_RANGES['T2b'][0]])
PARAM_HI = np.array([PARAM_RANGES['R1a'][1], PARAM_RANGES['M0b'][1], PARAM_RANGES['R'][1], PARAM_RANGES['T2b'][1]])
# Log-space bounds for relative-error-aware normalization
LOG_PARAM_LO = np.log(PARAM_LO)
LOG_PARAM_HI = np.log(PARAM_HI)
# =============
# FORWARD MODEL (signal simulation)
# =============

def compute_zspectrum_batch(R1a, R1b, M0b, R, T2a, T2b, Td, offsets_hz, w1_rad):
    """Steady-state Z-spectrum (Morrison-Henkelman two-pool model with dipolar order,
    Lorentzian free-water and Gaussian semisolid lineshapes).

    Parameter inputs are 1-D arrays of length N. offsets_hz is (n_offsets,) or (N, n_offsets);
    w1_rad is (n_B1,) or (N, n_B1). Returns (N, n_offsets * n_B1), flattened from
    shape (N, n_offsets, n_B1).
    """
    N = len(R1a)

    # Reshape arrays for numpy for faster compute with vector
    R1a, R1b, M0b, R = R1a[:,None,None], R1b[:,None,None], M0b[:,None,None], R[:,None,None]
    T2a, T2b, Td = T2a[:,None,None], T2b[:,None,None], Td[:,None,None]
    delta = (offsets_hz * 2 * np.pi)[:,:,None]  # (N, 52, 1) for per-sample B0 jitter
    # w1_rad can be (5,) for uniform or (N,5) for per-sample B1 variation
    if w1_rad.ndim == 1:
        w1 = w1_rad[None,None,:]
    else:
        w1 = w1_rad[:,None,:]  # (N, 1, 5)

    # RF saturation rates for two-pool MT model.
    rrfa = T2a * (w1**2) / (1 + (T2a * delta)**2)  # free water, Lorentzian lineshape
    rrfb = T2b * np.sqrt(np.pi/2) * (w1**2) * np.exp(-0.5 * (delta * T2b)**2) # semisolid pool, Gaussian lineshape

    # Morrison parameters
    S = R * M0b / R1a * R1b + R1b + R
    T = S + rrfa / R1a * (R1b + R)
    U = R * M0b / R1a + 1 + rrfa / R1a

    # Dipolar
    d = 1 / (np.sqrt(15) * T2b)
    dipolar = ((delta / d)**2) * rrfb * Td

    Z = (S * (1 + dipolar) + rrfb) / (T * (1 + dipolar) + rrfb * U)
    return np.clip(Z, 0, 1).reshape(N, -1)  # (N, 52*5=260)

def compute_selir_batch(R1a, R1b, M0b, R, Ma0, ti_times):
    """Selective inversion-recovery curve (two-pool biexponential recovery).

    Ma0 is the normalized water magnetization right after inversion (-1 = perfect).
    The curve is normalized so the last TI point equals 1. Returns (N, n_TI).
    """
    kab, kba = M0b * R, R
    sum_term = R1a + R1b + kab + kba
    diff_term = np.sqrt((R1a - R1b + kab - kba)**2 + 4 * kab * kba)
    R1p, R1n = 0.5 * (sum_term + diff_term), 0.5 * (sum_term - diff_term)

    denom = np.where(np.abs(R1p - R1n) < 1e-10, 1e-10, R1p - R1n)
    Bp = (R1a + kab - R1n) / denom
    Bn = -(R1a + kab - R1p) / denom

    t = ti_times[None, :]
    # Ma0 accounts for imperfect inversion (Ma0 = -1 for perfect, ~-0.85 to -0.97 experimentally)
    signal = 1 + (Ma0[:,None] - 1) * (Bp[:,None] * np.exp(-R1p[:,None] * t) + Bn[:,None] * np.exp(-R1n[:,None] * t))
    # Normalize to last TI point to match experimental data processing (last point = 1.0)
    signal = signal / signal[:, -1:]
    return signal # Shape is (number of samples, number of inversion time points)


# ==========================
# DATA GENERATION
# =========================

def generate_dataset(n_samples=N_SAMPLES, n_augment=N_AUGMENT, noise_std=NOISE_STD):
    """Simulate the synthetic training set.

    Draws n_samples target-parameter sets (log-uniform within PARAM_RANGES) and repeats
    each one n_augment times with independently drawn nuisance parameters (T2a, Td, Ma0,
    B0 shift, B1 scale), then adds Gaussian noise.

    Returns X (N, 285) = [Z-spectrum 260, SelIR 24, last-TI flag 1],
            Y (N, 4)   = [R1a, M0b, R, T2b],
            Nuisance   (only used by the optional consistency loss).
    """
    start_time = time.time()
    total = n_samples * n_augment
    print(f"Generating {n_samples:,} × {n_augment} = {total:,} samples...")

    # Base parameters (log-uniform), drawn from PARAM_RANGES
    R1a = np.repeat(np.exp(np.random.uniform(np.log(PARAM_RANGES['R1a'][0]), np.log(PARAM_RANGES['R1a'][1]), n_samples)), n_augment)
    M0b = np.repeat(np.exp(np.random.uniform(np.log(PARAM_RANGES['M0b'][0]), np.log(PARAM_RANGES['M0b'][1]), n_samples)), n_augment)
    R   = np.repeat(np.exp(np.random.uniform(np.log(PARAM_RANGES['R'][0]),   np.log(PARAM_RANGES['R'][1]),   n_samples)), n_augment)
    T2b = np.repeat(np.exp(np.random.uniform(np.log(PARAM_RANGES['T2b'][0]), np.log(PARAM_RANGES['T2b'][1]), n_samples)), n_augment)

    # Nuisance parameters, drawn from NUISANCE_RANGES
    R1b = np.random.uniform(*NUISANCE_RANGES['R1b'], total)
    # T2a spans ~2 orders of magnitude, so sample log-uniformly for even coverage
    T2a = np.exp(np.random.uniform(np.log(NUISANCE_RANGES['T2a'][0]),
                                    np.log(NUISANCE_RANGES['T2a'][1]), total))
    # Td spans ~20x range, so sample log-uniformly for even coverage
    Td  = np.exp(np.random.uniform(np.log(NUISANCE_RANGES['Td'][0]),
                                    np.log(NUISANCE_RANGES['Td'][1]), total))
    # Ma0 ceiling-correlated with M0b: higher macromolecular fraction → poorer inversion efficiency.
    # Ceiling goes from -0.85 (low M0b) up to -0.70 (high M0b=0.2), floor always -1.0.
    m0b_norm = (M0b - PARAM_RANGES['M0b'][0]) / (PARAM_RANGES['M0b'][1] - PARAM_RANGES['M0b'][0])
    ma0_ceiling = -0.85 + m0b_norm * 0.15  # -0.85 at M0b=0.001, -0.70 at M0b=0.2
    Ma0 = np.array([np.random.uniform(-1.0, c) for c in ma0_ceiling])

    # Process in batches
    batch_size = GEN_BATCH_SIZE
    X_list, Y_list, N_list = [], [], []

    # Simulate in batches to limit memory use
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        idx = slice(i, end)

        # B0 jitter, drawn from NUISANCE_RANGES
        b0_shift = np.random.uniform(*NUISANCE_RANGES['B0_jitter_hz'], end - i)
        offsets_shifted = OFFSETS_HZ + b0_shift[:, None]  # (batch, 52)

        # B1 inhomogeneity: scale RF power per sample
        b1_scale = np.random.uniform(*NUISANCE_RANGES['B1_scale'], end - i)
        w1_scaled = W1_RAD[None, :] * b1_scale[:, None]  # (batch, 5)

        zspec = compute_zspectrum_batch(R1a[idx], R1b[idx], M0b[idx], R[idx],
                                        T2a[idx], T2b[idx], Td[idx], offsets_shifted, w1_scaled)

        # Randomly assign last TI (10s or 13s) per sample to match experimental variety
        last_ti_vals = np.random.choice(TI_LAST_OPTIONS, size=(end - i))
        mask_10 = last_ti_vals == 10.0
        mask_13 = last_ti_vals == 13.0
        ti_13 = TI_TIMES.copy(); ti_13[-1] = 13.0
        selir = np.zeros((end - i, len(TI_TIMES)))
        if mask_10.any():
            selir[mask_10] = compute_selir_batch(R1a[idx][mask_10], R1b[idx][mask_10],
                                                 M0b[idx][mask_10], R[idx][mask_10],
                                                 Ma0[idx][mask_10], TI_TIMES)
        if mask_13.any():
            selir[mask_13] = compute_selir_batch(R1a[idx][mask_13], R1b[idx][mask_13],
                                                 M0b[idx][mask_13], R[idx][mask_13],
                                                 Ma0[idx][mask_13], ti_13)

        # Normalize last TI to [0, 1] and append as extra feature (285th input)
        last_ti_norm = (last_ti_vals - TI_LAST_MIN) / (TI_LAST_MAX - TI_LAST_MIN)

        # Heteroscedastic noise on selIR: early TI points get lower noise to match
        # the early-inversion weighting w = k/t + 1 used in fitting (10 <= k <= 50 ms).
        # Lower noise at short TI = higher effective SNR = NN learns to explain early
        # inversion recovery precisely, consistent with the weighted fitting ground truth.
        k_sec = np.random.uniform(0.010, 0.050, (end - i, 1))  # nuisance k per sample (s)
        w_ti = k_sec / TI_TIMES[None, :] + 1  # (batch, 24), large at short TI
        selir_noise = np.random.randn(*selir.shape) * (noise_std / w_ti)

        zspec_noise = np.random.randn(*zspec.shape) * noise_std
        signals = np.hstack([zspec + zspec_noise, selir + selir_noise])
        signals = np.hstack([signals, last_ti_norm[:, None]])   # TI feature added after noise

        X_list.append(signals.astype(np.float32))
        Y_list.append(np.stack([R1a[idx], M0b[idx], R[idx], T2b[idx]], axis=1).astype(np.float32))
        # Save nuisance params for forward model consistency loss
        N_list.append(np.stack([T2a[idx], Td[idx], Ma0[idx], last_ti_norm, b0_shift, b1_scale], axis=1).astype(np.float32))

        if (i + batch_size) % 100000 == 0 or end == total:
            print(f"  Progress: {end:,}/{total:,}")

    X, Y, N = np.vstack(X_list), np.vstack(Y_list), np.vstack(N_list)
    print(f"Generated: X={X.shape}, Y={Y.shape}, Nuisance={N.shape}")
    print(f"Data generation time: {time.time() - start_time:.1f}s")
    return X, Y, N

def normalize_params(Y):
    """Log-space normalization: equal MSE in this space ≈ equal relative error."""
    return (np.log(np.maximum(Y, PARAM_LO)) - LOG_PARAM_LO) / (LOG_PARAM_HI - LOG_PARAM_LO)

def denormalize_params(Y_norm):
    """Inverse of log-space normalization."""
    log_vals = np.clip(Y_norm, 0, 1) * (LOG_PARAM_HI - LOG_PARAM_LO) + LOG_PARAM_LO
    return np.clip(np.exp(log_vals), PARAM_LO, PARAM_HI)


# ===================
# PYTORCH FORWARD MODEL (for consistency loss)
# Only affects training when CONS_LAMBDA > 0. The manuscript used CONS_LAMBDA = 0.
# ===================

# Pre-convert constants to tensors (done once, moved to device in train_model)
_OFFSETS_RAD = torch.tensor(OFFSETS_HZ * 2 * np.pi, dtype=torch.float32)  # (52,)
_W1_RAD = torch.tensor(W1_RAD, dtype=torch.float32)  # (5,)
_TI_10 = torch.tensor(TI_TIMES, dtype=torch.float32)  # (24,)
_TI_13 = torch.tensor(np.concatenate([TI_TIMES[:-1], [13.0]]), dtype=torch.float32)

def forward_model_torch(R1a, M0b, R, T2b, T2a, Td, Ma0, last_ti_norm, offsets_rad, w1, ti_10, ti_13, b0_jitter=None, b1_scale=None):
    """Differentiable forward model: returns Z-spectrum (260) + selIR (24) = 284 features."""
    N = R1a.shape[0]
    R1b = 1.0

    # Reshape for broadcasting: (N,1,1) x (1,52,1) x (1,1,5)
    R1a_r = R1a.view(N,1,1); M0b_r = M0b.view(N,1,1); R_r = R.view(N,1,1)
    T2a_r = T2a.view(N,1,1); T2b_r = T2b.view(N,1,1); Td_r = Td.view(N,1,1)
    # Apply per-sample B0 jitter (Hz → rad/s) and B1 scaling if provided
    if b0_jitter is not None:
        delta = offsets_rad.view(1,-1,1) + b0_jitter.view(N,1,1) * (2 * np.pi)
    else:
        delta = offsets_rad.view(1,-1,1)  # (1,52,1)
    if b1_scale is not None:
        w1_r = w1.view(1,1,-1) * b1_scale.view(N,1,1)
    else:
        w1_r = w1.view(1,1,-1)  # (1,1,5)

    # RF saturation rates
    rrfa = T2a_r * (w1_r**2) / (1 + (T2a_r * delta)**2)
    rrfb = T2b_r * (np.pi/2)**0.5 * (w1_r**2) * torch.exp(-0.5 * (delta * T2b_r)**2)

    # Morrison parameters
    S = R_r * M0b_r / R1a_r * R1b + R1b + R_r
    T = S + rrfa / R1a_r * (R1b + R_r)
    U = R_r * M0b_r / R1a_r + 1 + rrfa / R1a_r

    # Dipolar
    d = 1.0 / (15**0.5 * T2b_r)
    dipolar = ((delta / d)**2) * rrfb * Td_r
    Z = (S * (1 + dipolar) + rrfb) / (T * (1 + dipolar) + rrfb * U)
    zspec = Z.clamp(0, 1).reshape(N, -1)  # (N, 260)

    # selIR — pick TI based on last_ti_norm (0=10s, 1=13s)
    # Use 10s TI for all, then correct last point for 13s samples
    Ma0_r = Ma0.view(N, 1)
    kab = M0b * R; kba = R
    sum_term = R1a + R1b + kab + kba
    diff_sq = (R1a - R1b + kab - kba)**2 + 4 * kab * kba
    diff_term = torch.sqrt(diff_sq.clamp(min=1e-20))
    R1p = 0.5 * (sum_term + diff_term)
    R1n = 0.5 * (sum_term - diff_term)
    denom = (R1p - R1n).clamp(min=1e-10)
    Bp = ((R1a + kab - R1n) / denom).view(N, 1)
    Bn = (-(R1a + kab - R1p) / denom).view(N, 1)
    R1p_r = R1p.view(N, 1); R1n_r = R1n.view(N, 1)

    # Compute for both TI sets, blend based on last_ti_norm
    t10 = ti_10.view(1, -1)
    t13 = ti_13.view(1, -1)
    sig10 = 1 + (Ma0_r - 1) * (Bp * torch.exp(-R1p_r * t10) + Bn * torch.exp(-R1n_r * t10))
    sig13 = 1 + (Ma0_r - 1) * (Bp * torch.exp(-R1p_r * t13) + Bn * torch.exp(-R1n_r * t13))
    sig10 = sig10 / sig10[:, -1:]
    sig13 = sig13 / sig13[:, -1:]

    # Blend: last_ti_norm=0 → use sig10, last_ti_norm=1 → use sig13
    w = last_ti_norm.view(N, 1)
    selir = (1 - w) * sig10 + w * sig13

    return torch.cat([zspec, selir], dim=1)  # (N, 284)


# ===================
# NEURAL NETWORK
# =================

class DRONE(nn.Module):
    """Fully connected network: 285 inputs -> 300 x 2 hidden (Tanh) -> 4 sigmoid outputs.

    Inputs: 260 Z-spectrum points (52 offsets x 5 B1), 24 SelIR points, 1 last-TI flag
    (0 = last TI 10 s, 1 = last TI 13 s). Outputs: normalized [R1a, M0b, R, T2b].
    """
    def __init__(self, input_dim=285, output_dim=4, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # Constrain output to [0,1]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ================
# TRAINING
# ================

def train_model(X, Y, N=None, epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE, lr=LEARNING_RATE):
    """Train DRONE on the simulated data.

    Splits 80/10/10 (train/val/test), trains with AdamW on a weighted MSE in
    log-normalized parameter space, keeps the best-validation checkpoint, and prints
    the mean % error per parameter on the synthetic test set.
    """
    start_time = time.time()
    _ckpt = f'best_model_{os.getpid()}.pt'  # per-process to avoid race with parallel TD runs
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # CUDA-only speedups (ignored on CPU / Apple MPS)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # Auto-tune for hardware
        torch.set_float32_matmul_precision('high')  # Use TensorCores

    Y_norm = normalize_params(Y)

    # Split: 80% train, 10% val, 10% test
    n = len(X)
    idx = np.random.permutation(n)
    test_size = int(n * 0.1)
    val_size = int(n * 0.1)
    test_idx = idx[:test_size]
    val_idx = idx[test_size:test_size + val_size]
    train_idx = idx[test_size + val_size:]

    # Load entire dataset to GPU (float32 required for MPS)
    X_train = torch.tensor(X[train_idx], device=device, dtype=torch.float32)
    Y_train = torch.tensor(Y_norm[train_idx], device=device, dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], device=device, dtype=torch.float32)
    Y_val = torch.tensor(Y_norm[val_idx], device=device, dtype=torch.float32)
    X_test = torch.tensor(X[test_idx], device=device, dtype=torch.float32)
    Y_test = torch.tensor(Y_norm[test_idx], device=device, dtype=torch.float32)

    # Nuisance params for consistency loss
    if N is not None:
        N_train = torch.tensor(N[train_idx], device=device, dtype=torch.float32)
        offsets_rad_t = _OFFSETS_RAD.to(device)
        w1_t = _W1_RAD.to(device)
        ti_10_t = _TI_10.to(device)
        ti_13_t = _TI_13.to(device)
        log_lo_t = torch.tensor(LOG_PARAM_LO, device=device, dtype=torch.float32)
        log_range_t = torch.tensor(LOG_PARAM_HI - LOG_PARAM_LO, device=device, dtype=torch.float32)

    model = DRONE().to(device)
    if device.type == 'cuda' and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')
    # Weight loss by log-range^2 so equal normalized MSE → equal percentage error
    log_ranges = torch.tensor(LOG_PARAM_HI - LOG_PARAM_LO, device=device, dtype=torch.float32)
    loss_w = (log_ranges ** 2)
    loss_w = loss_w / loss_w.mean()
    def criterion(pred, target):
        return (loss_w * (pred - target) ** 2).mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=(device.type == 'cuda'))
    # Warmup + ReduceLROnPlateau: ramp LR linearly for first WARMUP_EPOCHS, then decay on plateau
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0/WARMUP_EPOCHS, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, min_lr=1e-5)

    # Mixed precision for faster training on GPU
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    n_train = len(train_idx)
    n_val   = len(val_idx)
    n_batches_train = n_train // batch_size  # drop_last equivalent
    n_batches_val   = (n_val + batch_size - 1) // batch_size

    def eval_loss(X_data, Y_data, n_batches):
        model.eval()
        loss = torch.tensor(0.0, device=device)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for i in range(n_batches):
                xb = X_data[i * batch_size:(i + 1) * batch_size]
                yb = Y_data[i * batch_size:(i + 1) * batch_size]
                loss += criterion(model(xb), yb) * len(xb)
        return (loss / len(X_data)).item()

    print(f"Training: {n_train:,}, Validation: {n_val:,}, Test: {len(test_idx):,}")
    print(f"Mixed precision: {use_amp}")

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        # Shuffle on GPU — eliminates CPU DataLoader overhead entirely
        perm = torch.randperm(n_train, device=device)
        X_shuf, Y_shuf = X_train[perm], Y_train[perm]
        if N is not None:
            N_shuf = N_train[perm]
        train_loss = torch.tensor(0.0, device=device)
        for i in range(n_batches_train):
            xb = X_shuf[i * batch_size:(i + 1) * batch_size]
            yb = Y_shuf[i * batch_size:(i + 1) * batch_size]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(xb)
                param_loss = criterion(pred, yb)

                if N is not None:
                    nb = N_shuf[i * batch_size:(i + 1) * batch_size]
                    # Denormalize predictions (differentiable path)
                    pred_log = pred.clamp(0, 1) * log_range_t + log_lo_t
                    pred_params = torch.exp(pred_log)
                    # Forward model with predicted params + stored nuisance params
                    pred_signal = forward_model_torch(
                        pred_params[:,0], pred_params[:,1], pred_params[:,2], pred_params[:,3],
                        nb[:,0], nb[:,1], nb[:,2], nb[:,3],
                        offsets_rad_t, w1_t, ti_10_t, ti_13_t, nb[:,4], nb[:,5])
                    cons_loss = ((pred_signal - xb[:, :284]) ** 2).mean()
                    lam = min(1.0, epoch / max(CONS_WARMUP, 1)) * CONS_LAMBDA
                    loss = param_loss + lam * cons_loss
                else:
                    loss = param_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.detach()
        train_loss = (train_loss / n_batches_train).item()

        val_loss = eval_loss(X_val, Y_val, n_batches_val)
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), _ckpt)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

    model.load_state_dict(torch.load(_ckpt, weights_only=True))
    os.remove(_ckpt)

    # Compute mean % error per parameter on test set
    model.eval()
    with torch.no_grad():
        Y_pred_norm = model(X_test).cpu().numpy()
    Y_pred = denormalize_params(Y_pred_norm)
    Y_true = denormalize_params(Y_norm[test_idx])
    pct_errors = np.abs(Y_pred - Y_true) / Y_true * 100
    mean_pct_err = np.mean(pct_errors, axis=0)
    print(f"\nTest Set Mean % Error:")
    print(f"  R1a:  {mean_pct_err[0]:.2f}%")
    print(f"  M0b:  {mean_pct_err[1]:.2f}%")
    print(f"  R:    {mean_pct_err[2]:.2f}%")
    print(f"  T2b:  {mean_pct_err[3]:.2f}%")
    print(f"  Avg:  {np.mean(mean_pct_err):.2f}%")

    training_time = time.time() - start_time
    print(f"\nTraining time: {training_time:.1f}s")

    return model, device, mean_pct_err, train_losses, val_losses, training_time


# ==================
# EXPERIMENTAL VALIDATION
# ==================

def load_experimental_data():
    """Load experimental signals with ground truth from
    Newest_fitting_fig2values_1side2sideSelIR_noTIwt_incl11,7T.csv
    (NOadjfac, Rb=1, 1+simul2+SIR fit at 6.5mT).
    """
    base = Path(__file__).parent / 'experimental_data'

    selir_mtf = sio.loadmat(base / 'selIR_data_MTF_Gelatin.mat')
    zspec_mtf = sio.loadmat(base / 'zspec_data_MTF_Gelatin.mat')
    selir_tres = sio.loadmat(base / 'selIR_TRES.mat')
    zspec_tres = sio.loadmat(base / 'zspec_TRES_250204.mat')

    mtf_keys = {
        'MTF10':    ('ZSpecMTF10pc',     'S_MTF10'),
        'MTF20':    ('ZSpecMTF20pc',     'S_MTF20'),
        'MTF30':    ('ZSpecMTF30pc_1',   'S_MTF30'),
        'MTF40':    ('ZSpecMTF40pc_1',   'S_MTF40'),
        'MTF50pH4': ('ZSpecMTF50pc_pH4', 'S_MTF50pH4'),
        'MTF100':   ('ZSpecMTF100pc',    'S_MTF100'),
    }
    tres_keys = {
        'TRES100': ('Z_TRES_250204', 'S_TRES100'),
    }

    # Ground truth: Newest_fitting_fig2values_1side2sideSelIR_noTIwt_incl11,7T.csv
    # (NOadjfac, Rb=1, 1+simul2+SIR fit at 6.5mT)
    # ci: 95% confidence intervals as (lower, upper) per param
    ground_truth = {
        'MTF10':    {'R1a': 0.418021628, 'M0b': 0.009410272, 'R': 16.31858327, 'T2b': 1.73e-05},
        'MTF20':    {'R1a': 0.438835733, 'M0b': 0.021880578, 'R': 20.76319057, 'T2b': 1.77e-05},
        'MTF30':    {'R1a': 0.48569792,  'M0b': 0.037562104, 'R': 26.02784759, 'T2b': 1.75e-05},
        'MTF40':    {'R1a': 0.481782758, 'M0b': 0.050374598, 'R': 31.05828597, 'T2b': 1.76e-05},
        'MTF50pH4': {'R1a': 0.531545589, 'M0b': 0.073790791, 'R': 29.24496887, 'T2b': 1.80e-05},
        'MTF100':   {'R1a': 0.6663553,   'M0b': 0.136465418, 'R': 41.0988062,  'T2b': 1.68e-05},
        'TRES100':  {'R1a': 0.491500,    'M0b': 0.063030,    'R': 32.118000,   'T2b': 1.79e-05},
    }
    ci = {
        'MTF10':    {'R1a': (0.40792532,  0.428117937), 'M0b': (0.008436863, 0.010383681),
                     'R': (14.00578214, 18.63138441), 'T2b': (1.68e-05, 1.77e-05)},
        'MTF20':    {'R1a': (0.425963561, 0.451707905), 'M0b': (0.020687034, 0.023074121),
                     'R': (18.84126142, 22.68511971), 'T2b': (1.74e-05, 1.80e-05)},
        'MTF30':    {'R1a': (0.472049491, 0.499346349), 'M0b': (0.03616498,  0.038959228),
                     'R': (24.05796563, 27.99772955), 'T2b': (1.73e-05, 1.77e-05)},
        'MTF40':    {'R1a': (0.464254967, 0.499310548), 'M0b': (0.048385816, 0.052363381),
                     'R': (28.15774789, 33.95882406), 'T2b': (1.74e-05, 1.78e-05)},
        'MTF50pH4': {'R1a': (0.50622976,  0.556861417), 'M0b': (0.070423474, 0.077158109),
                     'R': (25.68578934, 32.8041484),  'T2b': (1.77e-05, 1.82e-05)},
        'MTF100':   {'R1a': (0.626654223, 0.706056378), 'M0b': (0.130645289, 0.142285547),
                     'R': (34.74427252, 47.45333988), 'T2b': (1.66e-05, 1.70e-05)},
        'TRES100':  {'R1a': (0.476812, 0.518694), 'M0b': (0.057265, 0.062304),
                     'R': (37.776302, 47.013946), 'T2b': (1.80e-05, 2.00e-05)},
    }

    data = {}

    def last_ti_feature(last_ti_sec):
        return np.float32((last_ti_sec - TI_LAST_MIN) / (TI_LAST_MAX - TI_LAST_MIN))

    mtf_10to40_names = {'MTF10', 'MTF20', 'MTF30', 'MTF40'}

    for name, (zkey, skey) in mtf_keys.items():
        z = zspec_mtf[zkey].flatten()
        s = selir_mtf[skey].flatten()
        last_ti = 13.0 if name in mtf_10to40_names else 10.0
        feat = last_ti_feature(last_ti)
        data[name] = {'signal': np.concatenate([z, s, [feat]]).astype(np.float32),
                      'truth': ground_truth[name], 'ci': ci[name]}

    for name, (zkey, skey) in tres_keys.items():
        z = zspec_tres[zkey].flatten()
        s = selir_tres[skey].flatten()
        feat = last_ti_feature(13.0)
        data[name] = {'signal': np.concatenate([z, s, [feat]]).astype(np.float32),
                      'truth': ground_truth[name], 'ci': ci[name]}

    return data

def evaluate_experimental(model, device):
    """Evaluate on experimental data against ground truth from
    Newest_fitting_fig2values_1side2sideSelIR_noTIwt_incl11,7T.csv."""
    print("\n" + "="*60)
    print("EXPERIMENTAL VALIDATION")
    print("="*60)

    data = load_experimental_data()

    print(f"\n{'Sample':<10} {'Param':<5} {'Truth':>8} {'Pred':>8} {'Err%':>7}")
    print("-"*45)

    all_errors = []
    per_sample = []  # (name, param, truth, pred, err)
    for name, d in data.items():
        signal = torch.tensor(d['signal']).unsqueeze(0).to(device)
        model.eval()
        with torch.no_grad():
            pred = denormalize_params(model(signal).cpu().numpy())[0]

        for i, p in enumerate(['R1a', 'M0b', 'R', 'T2b']):
            t, pr = d['truth'][p], pred[i]
            err = 100 * abs(pr - t) / t if t else 0
            all_errors.append(err)
            per_sample.append((name, p, t, float(pr), err))
            if p == 'T2b':
                print(f"{name:<10} {p:<5} {t:>10.3e} {pr:>10.3e} {err:>6.1f}%")
            else:
                print(f"{name:<10} {p:<5} {t:>8.4f} {pr:>8.4f} {err:>6.1f}%")
        print()

    print(f"Mean error: {np.mean(all_errors):.1f}%")
    print(f"Max error:  {np.max(all_errors):.1f}%")
    ci_data = {name: d['ci'] for name, d in data.items()}
    return all_errors, per_sample, ci_data


# ==================
# RESULT LOGGING
# ==================

def log_run(train_errors, exp_errors, per_sample,
            train_losses, val_losses, n_samples, n_augment, training_time, description="", ci_data=None):
    """Save loss / comparison plots and a markdown summary to result_history/runN/."""
    from datetime import datetime

    # Determine next run number
    base = Path(__file__).parent / 'result_history'
    base.mkdir(exist_ok=True)
    existing = [d for d in base.iterdir() if d.is_dir() and d.name.startswith('run')]
    run_num = len(existing) + 1
    run_dir = base / f'run{run_num}'
    while run_dir.exists():
        run_num += 1
        run_dir = base / f'run{run_num}'
    run_dir.mkdir()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # --- training_loss.png ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.ylim(0, 0.050)
    plt.legend()
    plt.grid(True)
    plt.savefig(run_dir / 'training_loss.png', dpi=150)
    plt.close()

    y_ranges = {'R1a': (0.0, 1.1), 'M0b': (0.0, 0.22), 'R': (15, 75), 'T2b': (5e-6, 3e-5)}
    params = ['R1a', 'M0b', 'R', 'T2b']

    def _make_comparison_plot(per_sample, title_suffix, filename, ci_data=None):
        samples = list(dict.fromkeys(n for n, _, _, _, _ in per_sample))
        truth_vals = {p: [] for p in params}
        pred_vals  = {p: [] for p in params}
        for s in samples:
            for p in params:
                t, pr = next((t, pr) for name, param, t, pr, _ in per_sample if name == s and param == p)
                truth_vals[p].append(t)
                pred_vals[p].append(pr)
        param_titles = {
            'R1a': r'$R_{1a}$ (s$^{-1}$)',
            'M0b': r'$M_{0b}$',
            'R':   r'$R$ (s$^{-1}$)',
            'T2b': r'$T_{2b}$ (s)',
        }
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, p in zip(axes.flat, params):
            x = np.arange(len(samples))
            t_vals = truth_vals[p]
            if ci_data:
                lo_errs = [max(0.0, t_vals[i] - ci_data[s][p][0]) for i, s in enumerate(samples)]
                hi_errs = [max(0.0, ci_data[s][p][1] - t_vals[i]) for i, s in enumerate(samples)]
                ax.errorbar(x, t_vals, yerr=[lo_errs, hi_errs], fmt='o-',
                            label='Truth (95% CI)', color='steelblue',
                            capsize=4, capthick=1.2, elinewidth=1.2)
            else:
                ax.plot(x, t_vals, 'o-', label='Truth', color='steelblue')
            ax.plot(x, pred_vals[p], 's--', label='NN Prediction', color='tomato')
            ax.set_xticks(x)
            ax.set_xticklabels(samples, rotation=15, ha='right')
            ax.set_ylim(*y_ranges[p])
            ax.set_title(param_titles[p])
            ax.set_ylabel(param_titles[p])
            ax.legend()
            ax.grid(True)
        fig.suptitle('Steady-state Neural Network Prediction vs Ground Truth', fontsize=13)
        plt.tight_layout()
        plt.savefig(run_dir / filename, dpi=150)
        plt.close()

    def _make_error_plot(per_sample, title_suffix, filename):
        samples = list(dict.fromkeys(n for n, _, _, _, _ in per_sample))
        err_matrix = {p: [] for p in params}
        for s in samples:
            for p in params:
                val = next(err for name, param, _, _, err in per_sample if name == s and param == p)
                err_matrix[p].append(val)
        x = np.arange(len(samples))
        width = 0.2
        fig, ax = plt.subplots(figsize=(12, 6))
        for i, p in enumerate(params):
            ax.bar(x + i * width, err_matrix[p], width, label=p, edgecolor='black')
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(samples, rotation=15)
        ax.set_ylabel('Absolute % Error')
        ax.set_ylim(0, 100)
        ax.set_title('Steady-state Error per Sample per Parameter')
        ax.axhline(10, color='red', linestyle='--', linewidth=1.2)
        ax.grid(True, axis='y')
        ax.legend()
        plt.tight_layout()
        plt.savefig(run_dir / filename, dpi=150)
        plt.close()

    if per_sample:  # experimental plots only when experimental data was available
        # --- comparison plots ---
        _make_comparison_plot(per_sample, 'NOadjfac + simul2', 'comparison_noadjfac.png', ci_data)

        # --- error analysis plots ---
        _make_error_plot(per_sample, 'NOadjfac + simul2', 'error_analysis_noadjfac.png')

    # --- result_history.md ---
    lines = []
    lines.append(f"# Run {run_num} — {timestamp}\n")

    if description:
        lines.append("## What was tested")
        lines.append(f"{description}\n")

    lines.append("## Configuration\n")
    lines.append("**Param Ranges**\n")
    lines.append("| Param | Min | Max |")
    lines.append("|---|---|---|")
    for k, (lo, hi) in PARAM_RANGES.items():
        lines.append(f"| {k} | {lo} | {hi} |")
    lines.append("")

    lines.append("**Nuisance Ranges**\n")
    lines.append("| Param | Min | Max |")
    lines.append("|---|---|---|")
    for k, (lo, hi) in NUISANCE_RANGES.items():
        lines.append(f"| {k} | {lo} | {hi} |")
    lines.append("")

    lines.append("**Hyperparameters**\n")
    lines.append("| Setting | Value |")
    lines.append("|---|---|")
    lines.append(f"| Hidden dim | {HIDDEN_DIM} |")
    lines.append(f"| Hidden layers | {N_LAYERS} |")
    lines.append(f"| Learning rate | {LEARNING_RATE} |")
    lines.append(f"| Epochs | {EPOCHS} |")
    lines.append(f"| Batch size | {TRAIN_BATCH_SIZE} |")
    lines.append(f"| Samples | {n_samples:,} |")
    lines.append(f"| n_augment | {n_augment} |")
    lines.append(f"| Total samples | {n_samples * n_augment:,} |")
    lines.append(f"| Consistency lambda | {CONS_LAMBDA} |")
    lines.append(f"| Consistency warmup | {CONS_WARMUP} |")
    lines.append(f"| Final train loss | {train_losses[-1]:.6f} |")
    lines.append(f"| Final val loss | {val_losses[-1]:.6f} |")
    lines.append(f"| Training time | {int(training_time // 60)}m {training_time % 60:.1f}s |")
    lines.append("")

    lines.append("## Test Set Errors (Synthetic)\n")
    lines.append("| Param | Mean % Error |")
    lines.append("|---|---|")
    for name, err in zip(['R1a', 'M0b', 'R', 'T2b'], train_errors):
        lines.append(f"| {name} | {err:.2f}% |")
    lines.append(f"| **Avg** | **{np.mean(train_errors):.2f}%** |")
    lines.append("")

    def _validation_section(lines, label, exp_errors, per_sample):
        outliers = {('MTF10', 'T2b')}
        lines.append(f"## Experimental Validation — {label}\n")
        lines.append("| Sample | Param | Truth | Pred | Err% |")
        lines.append("|---|---|---|---|---|")
        for name, param, truth, pred, err in per_sample:
            if param == 'T2b':
                lines.append(f"| {name} | {param} | {truth:.3e} | {pred:.3e} | {err:.1f}% |")
            else:
                lines.append(f"| {name} | {param} | {truth:.4f} | {pred:.4f} | {err:.1f}% |")
        lines.append(f"\n**Mean error: {np.mean(exp_errors):.1f}%** | **Max error: {np.max(exp_errors):.1f}%**\n")
        lines.append("| Param | Mean Err% (all) | Mean Err% (excl. MTF10 T2b) |")
        lines.append("|---|---|---|")
        for p in ['R1a', 'M0b', 'R', 'T2b']:
            all_errs  = [err for name, param, _, _, err in per_sample if param == p]
            excl_errs = [err for name, param, _, _, err in per_sample if param == p
                         and (name, param) not in outliers]
            lines.append(f"| {p} | {np.mean(all_errs):.1f}% | {np.mean(excl_errs):.1f}% |")
        all_errs_flat  = [err for _, _, _, _, err in per_sample]
        excl_errs_flat = [err for name, param, _, _, err in per_sample
                          if (name, param) not in outliers]
        lines.append(f"| **Overall** | **{np.mean(all_errs_flat):.1f}%** | **{np.mean(excl_errs_flat):.1f}%** |")

    if per_sample:
        _validation_section(lines, 'NOadjfac + simul2', exp_errors, per_sample)

    with open(run_dir / 'result_history.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Results saved to result_history/run{run_num}/")


# Main: simulate -> train -> experimental validation (if data present) -> save results
if __name__ == '__main__':
    X, Y, N = generate_dataset()
    model, device, train_errors, train_losses, val_losses, training_time = train_model(X, Y, N)

    # Experimental validation needs the phantom data, which is not in the public repo.
    data_dir = Path(__file__).parent / 'experimental_data'
    if data_dir.exists():
        exp_errors, per_sample, ci_data = evaluate_experimental(model, device)
    else:
        print(f"\nNo experimental data found at {data_dir}; skipping experimental validation.")
        exp_errors, per_sample, ci_data = [], [], None

    log_run(train_errors, exp_errors, per_sample,
            train_losses, val_losses, N_SAMPLES, N_AUGMENT, training_time,
            ci_data=ci_data,
            description="DRONE(285→300×2 Tanh→4), T2b=[8-25μs], ReduceLROnPlateau, ceiling Ma0 [-1.0,-0.70], NN-only inference. "
            "Ground truth: Newest_fitting_fig2values_1side2sideSelIR_noTIwt_incl11,7T.csv (NOadjfac+simul2).")
