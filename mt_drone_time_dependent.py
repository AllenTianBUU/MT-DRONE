"""
MT-DRONE: time-dependent (TD) model
===================================
Deep-learning estimation of qMT parameters R1a, M0b, R and T2b from time-dependent
Z-spectra (single B1 = 500 Hz, 38 offsets, saturation durations Tsat = 1.0, 1.5, 2.0,
8.0 s) plus the 24-point SelIR curve, acquired at 6.5 mT. Training Z-spectra are simulated
with the 3x3 Bloch-McConnell equations including the dipolar order pool.

Running `python mt_drone_time_dependent.py` does the whole pipeline:
  1. validate_against_steady_state()  check that at very long Tsat the time-dependent
                                      model matches the steady-state formula (< 1e-4).
                                      This also seeds NumPy (seed 42).
  2. generate_dataset()       simulate 50,000 x 25 = 1.25M noisy training signals
  3. train_model()            train a 266 -> 300 -> 300 -> 4 network and print the mean
                              % error on the 10% synthetic test set
  4. evaluate_experimental()  compare predictions on the phantom data to the fitted
                              ground truth (skipped if the data folders are missing)
  5. log_run()                save plots and a summary to result_history_timedependent/runN/
  6. validate_against_experiment()  residual heatmaps of simulated vs measured Z-spectra
                                    (only if the data folders are present)

Note: Code was written in a hybrid way between me and LLMs. I came up with the structure
and plan, and LLMs helped write the functions. I reviewed the code afterward and ran tests
to check accuracy. This version of the MT-DRONE was specifically cleaned up for a public release to maximize 
reproducibility.
"""

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
    'T2b': (15e-6, 19e-6),
}

# Nuisance parameter ranges — adjust here to propagate everywhere
NUISANCE_RANGES = {
    'R1b':          (1,     1),
    'T2a':          (0.030, 2.500),
    'Td':           (0.005, 0.1),
    'Ma0':          (-1.0, -0.85),  # flat range; experimental Ma0 ≈ -0.876 to -0.966, all within [-1.0,-0.85]
    'B0_jitter_hz': (-30,   30),
    'B1_scale':     (0.95,  1.05),
}

# Hyperparameters — adjust here to propagate everywhere
HIDDEN_DIM        = 300    # nodes per hidden layer in ZSpecNet
N_LAYERS          = 2      # number of hidden layers in ZSpecNet
LEARNING_RATE     = 0.001
EPOCHS            = 70
WARMUP_EPOCHS     = 5
TRAIN_BATCH_SIZE  = 4096   # batch size during training
GEN_BATCH_SIZE    = 4096   # batch size during data generation
N_SAMPLES         = 50000   # number of base parameter sets
N_AUGMENT         = 25     # nuisance parameter augmentations per base sample
NOISE_STD         = 0.01   # Gaussian noise std added to simulated signals
CONS_LAMBDA       = 0.0    # weight for forward model consistency loss (0 = disabled)
CONS_WARMUP       = 20     # epochs to ramp up consistency loss from 0 → full

# Frequency offsets from time-dependent experimental data (38 offsets)
OFFSETS_HZ = np.array([-36000, -32000, -28000, -24000, -20000, -16000, -12000,
    -8000, -6000, -5000, -4000, -3500, -3000, -2500, -2000, -1500, -1000, -500,
    0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 8000,
    12000, 16000, 20000, 24000, 28000, 32000, 36000, 70000], dtype=np.float64)

# Single saturation amplitude used in the time-dependent experiment (B1 = 500 Hz)
W1_RAD = 500.0 * 2 * np.pi  # ≈ 3141.593 rad/s

# Saturation durations for time-dependent Z-spectrum (7 values)
TSAT_VALUES = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 8.0], dtype=np.float64)

# Experimental repetition time defaults
TR_SECONDS = 14.0  # default constant TR (MTF50/100)

# Per-sample TR schedules for experimental evaluation:
# MTF50/100: delays [12.9,12.8,12.5,12,11.5,11,5]s → ~constant 14s total
# MTF30/40/10/20: 11s post-acquisition delay → TR = Tsat + ~1s acquisition + 11s
TR_MTF50_100 = np.full(len(TSAT_VALUES), 14.0)         # (7,) constant
TR_MTF30_40  = TSAT_VALUES + 1.0 + 11.0                # (7,) ≈ [12.1,12.2,12.5,13.0,13.5,14.0,20.0] s

_N_OFFS = len(OFFSETS_HZ)          # 38
N_ZSPEC_FEATS = 7 * _N_OFFS        # 266: full Z-spec (all 7 Tsat × 38 offsets)
N_INPUT_FEATS = N_ZSPEC_FEATS      # 266 network inputs (same size as the full 7-Tsat Z-spectrum)

# TI times for SelIR (24 points) — used by compute_selir_batch utility
TI_TIMES = np.array([0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02,
    0.025, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0,
    5.0, 10.0], dtype=np.float64)

# ── Hybrid input feature layout (N_INPUT_FEATS = 266, NN structure unchanged) ──
# [0:24]   = SelIR curve (24 TI points, normalized to last TI = 1.0)
# [24]     = last_ti_norm  (0.0 = TI_last 10s,  1.0 = TI_last 13s)
# [25:114] = zeros (89 padding slots — weights converge to ~0 during training)
# [114:266]= Z-spectrum at Tsat = [1.0, 1.5, 2.0, 8.0] × 38 offsets = 152 features
#            (same positions as Tsat[3:7] in the original 7-Tsat full Z-spec layout)
N_SELIR_FEATS   = len(TI_TIMES)                      # 24
SELIR_ZERO_PAD  = 89                                 # padding [25:114]
TSAT_LONG       = TSAT_VALUES[3:]                    # [1.0, 1.5, 2.0, 8.0]
N_TSAT_LONG     = len(TSAT_LONG)                     # 4
N_LONG_ZSPEC    = N_TSAT_LONG * _N_OFFS              # 152  — occupies [114:266]

# Possible last TI values across experimental SelIR samples
TI_LAST_OPTIONS = np.array([10.0, 13.0], dtype=np.float64)
TI_LAST_MIN, TI_LAST_MAX = 10.0, 13.0

# Normalization bounds derived from PARAM_RANGES — do not edit directly
PARAM_LO = np.array([PARAM_RANGES['R1a'][0], PARAM_RANGES['M0b'][0], PARAM_RANGES['R'][0], PARAM_RANGES['T2b'][0]])
PARAM_HI = np.array([PARAM_RANGES['R1a'][1], PARAM_RANGES['M0b'][1], PARAM_RANGES['R'][1], PARAM_RANGES['T2b'][1]])
# Log-space bounds for relative-error-aware normalization
LOG_PARAM_LO = np.log(PARAM_LO)
LOG_PARAM_HI = np.log(PARAM_HI)


# =============
# Forward simulate (Bloch-McConnell ODE, time-dependent)
# =============

# Original steady-state Morrison formula (kept for validation)
def compute_zspectrum_steady_state(R1a, R1b, M0b, R, T2a, T2b, Td, offsets_hz, w1_rad):
    N = len(R1a)
    R1a, R1b, M0b, R = R1a[:,None,None], R1b[:,None,None], M0b[:,None,None], R[:,None,None]
    T2a, T2b, Td = T2a[:,None,None], T2b[:,None,None], Td[:,None,None]
    delta = (offsets_hz * 2 * np.pi)[:,:,None]
    if w1_rad.ndim == 1:
        w1 = w1_rad[None,None,:]
    else:
        w1 = w1_rad[:,None,:]
    rrfa = T2a * (w1**2) / (1 + (T2a * delta)**2)
    rrfb = T2b * np.sqrt(np.pi/2) * (w1**2) * np.exp(-0.5 * (delta * T2b)**2)
    S = R * M0b / R1a * R1b + R1b + R
    T = S + rrfa / R1a * (R1b + R)
    U = R * M0b / R1a + 1 + rrfa / R1a
    d = 1 / (np.sqrt(15) * T2b)
    dipolar = ((delta / d)**2) * rrfb * Td
    Z = (S * (1 + dipolar) + rrfb) / (T * (1 + dipolar) + rrfb * U)
    return np.clip(Z, 0, 1).reshape(N, -1)


# SelIR forward model (biexponential inversion recovery, identical to mt_drone.py)
def compute_selir_batch(R1a, R1b, M0b, R, Ma0, ti_times):
    kab, kba = M0b * R, R
    sum_term = R1a + R1b + kab + kba
    diff_term = np.sqrt((R1a - R1b + kab - kba)**2 + 4 * kab * kba)
    R1p, R1n = 0.5 * (sum_term + diff_term), 0.5 * (sum_term - diff_term)
    denom = np.where(np.abs(R1p - R1n) < 1e-10, 1e-10, R1p - R1n)
    Bp = (R1a + kab - R1n) / denom
    Bn = -(R1a + kab - R1p) / denom
    t = ti_times[None, :]
    signal = 1 + (Ma0[:,None] - 1) * (Bp[:,None] * np.exp(-R1p[:,None] * t) + Bn[:,None] * np.exp(-R1n[:,None] * t))
    signal = signal / signal[:, -1:]
    return signal  # (N, n_ti)


# Time-dependent Z-spectrum via 3x3 longitudinal Bloch-McConnell with explicit dipolar order
def compute_zspectrum_batch(R1a, R1b, M0b, R, T2a, T2b, Td, offsets_hz, w1_rad, t_sat,
                            tr=TR_SECONDS):
    """
    3x3 longitudinal Bloch-McConnell ODE with explicit dipolar order pool
    and periodic steady-state correction. Follows Morrison & Henkelman (MRM 1995).

    State vector: [Mza, Mzb, D] where D is the dipolar order of the semisolid pool.

    tr: repetition time. Supports:
      - scalar: same TR for all N samples and all Tsat
      - (n_tsat,): per-Tsat TR, same for all N (used for experimental evaluation)
      - (N,): per-sample TR, same for all Tsat (not used currently)
      - (N, n_tsat): per-sample, per-Tsat (used during training for TR randomization)
    """
    scalar_tsat = np.ndim(t_sat) == 0
    t_sat = np.atleast_1d(np.asarray(t_sat, dtype=np.float64))
    n_tsat = len(t_sat)
    N = len(R1a)
    n_offs = offsets_hz.shape[1]

    # Per-sample B1
    w1_arr = np.atleast_2d(np.asarray(w1_rad, dtype=np.float64))
    if w1_arr.shape[0] == 1:
        w1 = np.full(N, w1_arr.flat[0])
    else:
        w1 = w1_arr[:, 0]

    delta = offsets_hz * 2 * np.pi  # (N, n_offs) rad/s
    kab = M0b * R     # (N,)
    kba = R            # (N,)

    # Direct water saturation rate (Lorentzian lineshape)
    rrfa = T2a[:, None] * w1[:, None]**2 / (1 + (T2a[:, None] * delta)**2)

    # Semisolid saturation rate (Gaussian lineshape, bare — no dipolar correction)
    T2b_e = T2b[:, None]
    w1_e = w1[:, None]
    rrfb = T2b_e * np.sqrt(np.pi / 2) * w1_e**2 * np.exp(-0.5 * (delta * T2b_e)**2)

    # Dipolar coupling factor: α² = (Δ/d)² = 15·(Δ·T2b)²
    alpha_sq = 15.0 * (delta * T2b_e)**2  # (N, n_offs)

    # ── 3x3 saturation matrix A_sat ──
    s11 = -(R1a + kab)[:, None] - rrfa
    s12 = kba[:, None] * np.ones((1, n_offs))
    # s13 = 0
    s21 = kab[:, None] * np.ones((1, n_offs))
    s22 = -(R1b + kba)[:, None] - rrfb
    s23 = alpha_sq * rrfb
    # s31 = 0
    s32 = rrfb
    s33 = -(1.0 / Td[:, None]) - alpha_sq * rrfb

    # ── 2x2 recovery matrix A_rec (no RF, D decouples) ──
    r11 = -(R1a + kab)[:, None] * np.ones((1, n_offs))
    r12 = kba[:, None] * np.ones((1, n_offs))
    r21 = kab[:, None] * np.ones((1, n_offs))
    r22 = -(R1b + kba)[:, None] * np.ones((1, n_offs))

    # Driving vector b = [R1a, R1b·M0b, 0]
    b1 = R1a[:, None] * np.ones((1, n_offs))
    b2 = (R1b * M0b)[:, None] * np.ones((1, n_offs))

    # ── Saturation steady state via Cramer's rule (3x3) ──
    det_s = s11 * (s22 * s33 - s23 * s32) - s12 * s21 * s33
    mss_z = (-b1 * (s22 * s33 - s23 * s32) + s12 * b2 * s33) / det_s
    mss_b = s33 * (b1 * s21 - s11 * b2) / det_s
    mss_d = s32 * (s11 * b2 - b1 * s21) / det_s

    # ── Eigenvalues of 3x3 saturation matrix (analytical cubic) ──
    trace_A = s11 + s22 + s33
    minor_sum = (s11 * s22 - s12 * s21) + s33 * (s11 + s22) - s23 * s32
    a_poly = -trace_A
    b_poly = minor_sum
    c_poly = -det_s
    p_dep = b_poly - a_poly**2 / 3.0
    q_dep = c_poly - a_poly * b_poly / 3.0 + 2.0 * a_poly**3 / 27.0
    p_dep = np.minimum(p_dep, -1e-30)
    m_cubic = 2.0 * np.sqrt(-p_dep / 3.0)
    cos_arg = np.clip(3.0 * q_dep / (p_dep * m_cubic), -1.0, 1.0)
    theta = np.arccos(cos_arg) / 3.0
    shift = -a_poly / 3.0
    lam1 = m_cubic * np.cos(theta) + shift
    lam2 = m_cubic * np.cos(theta - 2 * np.pi / 3) + shift
    lam3 = m_cubic * np.cos(theta + 2 * np.pi / 3) + shift

    # ── Precompute A² elements (exploiting s13=s31=0) ──
    a2_11 = s11**2 + s12 * s21
    a2_12 = s12 * (s11 + s22)
    a2_13 = s12 * s23
    a2_21 = s21 * (s11 + s22)
    a2_22 = s21 * s12 + s22**2 + s23 * s32
    a2_23 = s23 * (s22 + s33)
    a2_31 = s32 * s21
    a2_32 = s32 * (s22 + s33)
    a2_33 = s32 * s23 + s33**2

    # ── 2x2 recovery helpers ──
    def eig2x2(a, b, c, d):
        tr = a + d
        det = a * d - b * c
        disc = np.maximum(tr**2 - 4 * det, 0.0)
        sq = np.sqrt(disc)
        return (tr + sq) / 2, (tr - sq) / 2

    def expm2x2(a, b, c, d, t, lam1, lam2):
        dl = lam1 - lam2
        e1 = np.exp(lam1 * t)
        e2 = np.exp(lam2 * t)
        c0 = (lam1 * e2 - lam2 * e1) / dl
        c1 = (e1 - e2) / dl
        return c0 + c1 * a, c1 * b, c1 * c, c0 + c1 * d

    lr1, lr2 = eig2x2(r11, r12, r21, r22)

    # Recovery equilibrium: [1, M0b, 0]
    meq_z = np.ones_like(s11)
    meq_b = M0b[:, None] * np.ones((1, n_offs))

    Z_all = np.zeros((N, n_offs, n_tsat))

    # Pre-process tr into a 2D array (N, n_tsat) for unified indexing
    tr_arr = np.asarray(tr, dtype=np.float64)
    if tr_arr.ndim == 0:
        tr_2d = np.full((N, n_tsat), float(tr_arr))
    elif tr_arr.ndim == 1 and len(tr_arr) == n_tsat:
        tr_2d = np.tile(tr_arr[None, :], (N, 1))
    elif tr_arr.ndim == 1 and len(tr_arr) == N:
        tr_2d = np.tile(tr_arr[:, None], (1, n_tsat))
    else:  # (N, n_tsat)
        tr_2d = tr_arr

    for ti, ts in enumerate(t_sat):
        t_rec = tr_2d[:, ti:ti+1] - ts  # (N, 1), broadcasts over n_offs

        # ── Saturation propagator P via Cayley-Hamilton: exp(At) = c0·I + c1·A + c2·A² ──
        e1 = np.exp(lam1 * ts)
        e2 = np.exp(lam2 * ts)
        e3 = np.exp(lam3 * ts)

        d12 = lam1 - lam2
        d13 = lam1 - lam3
        d23 = lam2 - lam3
        eps = 1e-30
        d12 = np.where(np.abs(d12) < eps, eps, d12)
        d13 = np.where(np.abs(d13) < eps, eps, d13)
        d23 = np.where(np.abs(d23) < eps, eps, d23)

        f1 = 1.0 / (d12 * d13)
        f2 = 1.0 / (-d12 * d23)
        f3 = 1.0 / (d13 * d23)

        cc2 = f1 * e1 + f2 * e2 + f3 * e3
        cc1 = -(f1 * (lam2 + lam3) * e1 + f2 * (lam1 + lam3) * e2
              + f3 * (lam1 + lam2) * e3)
        cc0 = (f1 * lam2 * lam3 * e1 + f2 * lam1 * lam3 * e2
             + f3 * lam1 * lam2 * e3)

        P11 = cc0 + cc1 * s11 + cc2 * a2_11
        P12 = cc1 * s12 + cc2 * a2_12
        P13 = cc2 * a2_13
        P21 = cc1 * s21 + cc2 * a2_21
        P22 = cc0 + cc1 * s22 + cc2 * a2_22
        P23 = cc1 * s23 + cc2 * a2_23
        P31 = cc2 * a2_31
        P32 = cc1 * s32 + cc2 * a2_32
        P33 = cc0 + cc1 * s33 + cc2 * a2_33

        # ── Recovery propagator Q (block diagonal: 2x2 + scalar decay) ──
        Q11, Q12, Q21, Q22 = expm2x2(r11, r12, r21, r22, t_rec, lr1, lr2)
        Q33 = np.exp(-t_rec / Td[:, None])

        # ── QP product ──
        QP11 = Q11 * P11 + Q12 * P21
        QP12 = Q11 * P12 + Q12 * P22
        QP13 = Q11 * P13 + Q12 * P23
        QP21 = Q21 * P11 + Q22 * P21
        QP22 = Q21 * P12 + Q22 * P22
        QP23 = Q21 * P13 + Q22 * P23
        QP31 = Q33 * P31
        QP32 = Q33 * P32
        QP33 = Q33 * P33

        # ── RHS: Q·(I-P)·Mss + (I-Q)·Meq ──
        IPz = (1 - P11) * mss_z - P12 * mss_b - P13 * mss_d
        IPb = -P21 * mss_z + (1 - P22) * mss_b - P23 * mss_d
        IPd = -P31 * mss_z - P32 * mss_b + (1 - P33) * mss_d
        QIPz = Q11 * IPz + Q12 * IPb
        QIPb = Q21 * IPz + Q22 * IPb
        QIPd = Q33 * IPd
        IQz = (1 - Q11) * meq_z - Q12 * meq_b
        IQb = -Q21 * meq_z + (1 - Q22) * meq_b

        rz = QIPz + IQz
        rb = QIPb + IQb
        rd = QIPd

        # ── Solve (I - QP)·M0 = rhs via Cramer's rule (3x3) ──
        J11 = 1 - QP11;  J12 = -QP12;   J13 = -QP13
        J21 = -QP21;     J22 = 1 - QP22; J23 = -QP23
        J31 = -QP31;     J32 = -QP32;    J33 = 1 - QP33

        det_J = (J11 * (J22 * J33 - J23 * J32)
               - J12 * (J21 * J33 - J23 * J31)
               + J13 * (J21 * J32 - J22 * J31))

        M0z = (rz * (J22 * J33 - J23 * J32)
             - J12 * (rb * J33 - J23 * rd)
             + J13 * (rb * J32 - J22 * rd)) / det_J

        dz = M0z - mss_z
        db = (J11 * (rb * J33 - J23 * rd)
               - rz * (J21 * J33 - J23 * J31)
               + J13 * (J21 * rd - rb * J31)) / det_J - mss_b
        dd = (J11 * (J22 * rd - rb * J32)
             - J12 * (J21 * rd - rb * J31)
             + rz * (J21 * J32 - J22 * J31)) / det_J - mss_d

        Z_z = P11 * dz + P12 * db + P13 * dd + mss_z

        Z_all[:, :, ti] = Z_z

    Z_all = np.clip(Z_all, 0, 1)

    if scalar_tsat:
        return Z_all[:, :, 0]
    else:
        return Z_all.transpose(0, 2, 1).reshape(N, -1)


# Validate time-dependent matches steady-state at large t_sat
def validate_against_steady_state(n_test=1000, t_sat_large=200.0, atol=1e-4):
    np.random.seed(42)
    R1a = np.random.uniform(*PARAM_RANGES['R1a'], n_test)
    M0b = np.random.uniform(*PARAM_RANGES['M0b'], n_test)
    R   = np.random.uniform(*PARAM_RANGES['R'], n_test)
    T2b = np.random.uniform(*PARAM_RANGES['T2b'], n_test)
    R1b = np.ones(n_test)
    T2a = np.random.uniform(*NUISANCE_RANGES['T2a'], n_test)
    Td  = np.random.uniform(*NUISANCE_RANGES['Td'], n_test)

    offsets = np.tile(OFFSETS_HZ, (n_test, 1))
    w1_arr = np.array([W1_RAD])

    Z_ss = compute_zspectrum_steady_state(R1a, R1b, M0b, R, T2a, T2b, Td, offsets, w1_arr)
    Z_td = compute_zspectrum_batch(R1a, R1b, M0b, R, T2a, T2b, Td, offsets, w1_arr,
                                    t_sat=t_sat_large, tr=t_sat_large + 100.0)

    max_diff = np.max(np.abs(Z_td - Z_ss))
    mean_diff = np.mean(np.abs(Z_td - Z_ss))
    print(f"Validation: max diff = {max_diff:.2e}, mean diff = {mean_diff:.2e}")
    assert max_diff < atol, f"FAILED: max diff {max_diff:.2e} > tolerance {atol}"
    print("PASSED: Time-dependent matches steady-state at large t_sat")


# ==========================
# DATA GENERATION
# =========================

def generate_dataset(n_samples=N_SAMPLES, n_augment=N_AUGMENT, noise_std=NOISE_STD):
    """Simulate the synthetic training set.

    Draws n_samples target-parameter sets (log-uniform within PARAM_RANGES) and repeats
    each one n_augment times with independently drawn nuisance parameters (T2a, Td, Ma0,
    B0 shift, B1 scale, TR schedule), then adds Gaussian noise.

    Returns X (N, 266) laid out as in "Hybrid input feature layout" above,
            Y (N, 4) = [R1a, M0b, R, T2b],
            Nuisance (only used by the optional consistency loss).
    """
    start_time = time.time()
    total = n_samples * n_augment
    print(f"Generating {n_samples:,} × {n_augment} = {total:,} samples...")

    # Base parameters (log-uniform), drawn from PARAM_RANGES
    R1a = np.repeat(np.exp(np.random.uniform(np.log(PARAM_RANGES['R1a'][0]), np.log(PARAM_RANGES['R1a'][1]), n_samples)), n_augment)
    M0b = np.repeat(np.exp(np.random.uniform(np.log(PARAM_RANGES['M0b'][0]), np.log(PARAM_RANGES['M0b'][1]), n_samples)), n_augment)
    R   = np.repeat(np.exp(np.random.uniform(np.log(PARAM_RANGES['R'][0]),   np.log(PARAM_RANGES['R'][1]),   n_samples)), n_augment)
    T2b = np.repeat(np.exp(np.random.uniform(np.log(PARAM_RANGES['T2b'][0]), np.log(PARAM_RANGES['T2b'][1]), n_samples)), n_augment)

    # Nuisance parameters
    R1b = np.random.uniform(*NUISANCE_RANGES['R1b'], total)
    T2a = np.exp(np.random.uniform(np.log(NUISANCE_RANGES['T2a'][0]), np.log(NUISANCE_RANGES['T2a'][1]), total))
    Td  = np.exp(np.random.uniform(np.log(NUISANCE_RANGES['Td'][0]),  np.log(NUISANCE_RANGES['Td'][1]),  total))
    # Ma0: flat range — all experimental Ma0 values are within [-1.0, -0.85]
    Ma0 = np.random.uniform(*NUISANCE_RANGES['Ma0'], total)

    batch_size = GEN_BATCH_SIZE
    X_list, Y_list, N_list = [], [], []
    # TI array for 13s last-TI variant (MTF10–40 experimental protocol)
    _ti_13 = np.concatenate([TI_TIMES[:-1], [13.0]])

    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        idx = slice(i, end)
        bsize = end - i

        # B0/B1 nuisances
        b0_shift = np.random.uniform(*NUISANCE_RANGES['B0_jitter_hz'], bsize)
        offsets_shifted = OFFSETS_HZ + b0_shift[:, None]  # (batch, 38)
        b1_scale = np.random.uniform(*NUISANCE_RANGES['B1_scale'], bsize)
        w1_batch = np.array([W1_RAD]) * b1_scale[:, None]  # (batch, 1)

        # ── SelIR (24 features): randomly assign last TI = 10s or 13s per sample ──
        last_ti_vals = np.random.choice(TI_LAST_OPTIONS, size=bsize)
        mask_13 = last_ti_vals == 13.0
        selir = np.zeros((bsize, N_SELIR_FEATS), dtype=np.float64)
        if (~mask_13).any():
            selir[~mask_13] = compute_selir_batch(
                R1a[idx][~mask_13], R1b[idx][~mask_13],
                M0b[idx][~mask_13], R[idx][~mask_13],
                Ma0[idx][~mask_13], TI_TIMES)
        if mask_13.any():
            selir[mask_13] = compute_selir_batch(
                R1a[idx][mask_13], R1b[idx][mask_13],
                M0b[idx][mask_13], R[idx][mask_13],
                Ma0[idx][mask_13], _ti_13)
        last_ti_norm = (last_ti_vals - TI_LAST_MIN) / (TI_LAST_MAX - TI_LAST_MIN)  # 0.0 or 1.0

        # Heteroscedastic SelIR noise: less noise at short TI (higher SNR at early recovery)
        k_sec = np.random.uniform(0.010, 0.050, (bsize, 1))
        w_ti  = k_sec / TI_TIMES[None, :] + 1.0   # (bsize, 24), large at short TI
        selir_noisy = selir + np.random.randn(*selir.shape) * (noise_std / w_ti)

        # ── Z-spectrum at Tsat = [1.0, 1.5, 2.0, 8.0] s × 38 offsets = 152 features ──
        # Per-sample TR schedule: 50% constant 14s, 50% Tsat+1+d, d~U[5,15]s
        use_constant = np.random.rand(bsize) < 0.5
        d_delay = np.random.uniform(5.0, 15.0, bsize)
        tr_long = np.where(
            use_constant[:, None],
            14.0,
            TSAT_LONG[None, :] + 1.0 + d_delay[:, None]
        )  # (bsize, N_TSAT_LONG)

        zspec_long = compute_zspectrum_batch(
            R1a[idx], R1b[idx], M0b[idx], R[idx],
            T2a[idx], T2b[idx], Td[idx], offsets_shifted, w1_batch,
            TSAT_LONG, tr=tr_long)  # (bsize, N_LONG_ZSPEC)
        zspec_noisy = zspec_long + np.random.randn(*zspec_long.shape) * noise_std

        # ── Assemble 266-dim input: [SelIR(24), last_ti_norm(1), zeros(89), Z_4tsat(152)] ──
        signals = np.hstack([
            selir_noisy.astype(np.float32),            # [0:24]   SelIR
            last_ti_norm[:, None].astype(np.float32),  # [24]     last_ti_norm
            np.zeros((bsize, SELIR_ZERO_PAD), dtype=np.float32),  # [25:114] zeros
            zspec_noisy.astype(np.float32),            # [114:266] Z-spec 4 Tsat
        ])  # (bsize, 266)

        X_list.append(signals)
        Y_list.append(np.stack([R1a[idx], M0b[idx], R[idx], T2b[idx]], axis=1).astype(np.float32))
        N_list.append(np.stack([T2a[idx], Td[idx], b0_shift, b1_scale], axis=1).astype(np.float32))

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
# PYTORCH FORWARD MODEL (for consistency loss — 2x2 approximation, CONS_LAMBDA=0 so unused)
# ===================

_OFFSETS_RAD = torch.tensor(OFFSETS_HZ * 2 * np.pi, dtype=torch.float32)  # (38,)
_W1_RAD = torch.tensor([W1_RAD], dtype=torch.float32)  # (1,)
_TSAT_VALUES = torch.tensor(TSAT_VALUES, dtype=torch.float32)  # (7,)

def expm_2x2_batch_torch(a11, a12, a21, a22, t):
    tr = a11 + a22
    det = a11 * a22 - a12 * a21
    disc_sq = (tr**2 - 4 * det).clamp(min=0.0)
    disc = torch.sqrt(disc_sq)
    lam1 = 0.5 * (tr + disc)
    lam2 = 0.5 * (tr - disc)
    exp1 = torch.exp(lam1 * t)
    exp2 = torch.exp(lam2 * t)
    dlam = lam1 - lam2
    safe_dlam = torch.where(dlam.abs() < 1e-12, torch.ones_like(dlam), dlam)
    e11 = (exp1 * (a11 - lam2) - exp2 * (a11 - lam1)) / safe_dlam
    e12 = (exp1 - exp2) * a12 / safe_dlam
    e21 = (exp1 - exp2) * a21 / safe_dlam
    e22 = (exp1 * (a22 - lam2) - exp2 * (a22 - lam1)) / safe_dlam
    degen = dlam.abs() < 1e-12
    if degen.any():
        lam = 0.5 * tr
        exp_lt = torch.exp(lam * t)
        e11 = torch.where(degen, exp_lt * (1 + (a11 - lam) * t), e11)
        e12 = torch.where(degen, exp_lt * a12 * t, e12)
        e21 = torch.where(degen, exp_lt * a21 * t, e21)
        e22 = torch.where(degen, exp_lt * (1 + (a22 - lam) * t), e22)
    return e11, e12, e21, e22

def forward_model_torch(R1a, M0b, R, T2b, T2a, Td, offsets_rad, w1, tsat_values, b0_jitter=None, b1_scale=None):
    """Differentiable forward model: returns time-dependent Z-spectrum (38*7=266 features).
    Uses 2x2 steady-state dipolar approximation. Only used if CONS_LAMBDA > 0."""
    N = R1a.shape[0]
    R1b = 1.0

    R1a_r = R1a.view(N, 1, 1); M0b_r = M0b.view(N, 1, 1); R_r = R.view(N, 1, 1)
    T2a_r = T2a.view(N, 1, 1); T2b_r = T2b.view(N, 1, 1); Td_r = Td.view(N, 1, 1)

    if b0_jitter is not None:
        delta = offsets_rad.view(1, -1, 1) + b0_jitter.view(N, 1, 1) * (2 * np.pi)
    else:
        delta = offsets_rad.view(1, -1, 1)
    if b1_scale is not None:
        w1_r = w1.view(1, 1, -1) * b1_scale.view(N, 1, 1)
    else:
        w1_r = w1.view(1, 1, -1)

    kab = R_r * M0b_r
    kba = R_r

    rrfa = T2a_r * (w1_r**2) / (1 + (T2a_r * delta)**2)
    rrfb = T2b_r * (np.pi / 2)**0.5 * (w1_r**2) * torch.exp(-0.5 * (delta * T2b_r)**2)

    d = 1.0 / (15**0.5 * T2b_r)
    dipolar_term = ((delta / d)**2) * rrfb * Td_r
    rrfb_eff = rrfb / (1.0 + dipolar_term)

    a11 = -(R1a_r + kab + rrfa)
    a12 = kba
    a21 = kab
    a22 = -(R1b + kba + rrfb_eff)

    b1_vec = R1a_r
    b2_vec = R1b * M0b_r

    det_A = a11 * a22 - a12 * a21
    safe_det = torch.where(det_A.abs() < 1e-30, torch.tensor(1e-30, device=R1a.device), det_A)
    mss_a = -(a22 * b1_vec - a12 * b2_vec) / safe_det
    mss_b = -(-a21 * b1_vec + a11 * b2_vec) / safe_det

    M0a = 1.0
    dm_a = M0a - mss_a
    dm_b = M0b_r - mss_b

    z_list = []
    for t_sat in tsat_values:
        e11, e12, _, _ = expm_2x2_batch_torch(a11, a12, a21, a22, t_sat)
        mza_t = e11 * dm_a + e12 * dm_b + mss_a
        z = mza_t.clamp(0, 1).reshape(N, -1)
        z_list.append(z)

    return torch.cat(z_list, dim=1)  # (N, 38*7=266)


# ===================
# NEURAL NETWORK — DRONE
# =================

class DRONE(nn.Module):
    """
    Fully connected network: 266 inputs → 300×2 hidden (Tanh) → 4 sigmoid outputs.
    Input layout: see "Hybrid input feature layout" in SETTINGS
    (SelIR 24 + last-TI flag 1 + zero padding 89 + Z-spectrum 152).
    Output order: [R1a, M0b, R, T2b]
    """
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(N_INPUT_FEATS, HIDDEN_DIM), nn.Tanh()]
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Tanh()]
        layers += [nn.Linear(HIDDEN_DIM, 4), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)  # (N, 4): [R1a, M0b, R, T2b]


# ================
# TRAINING
# ================

def train_model(X, Y, N=None, epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE, lr=LEARNING_RATE):
    """Train DRONE on the simulated data.

    Splits 80/10/10 (train/val/test), trains with AdamW (linear warmup, then cosine
    annealing) on a weighted MSE in log-normalized parameter space, keeps the
    best-validation checkpoint, and prints the mean % error on the synthetic test set.
    """
    import os
    start_time = time.time()
    _ckpt = f'best_model_{os.getpid()}.pt'  # per-process to avoid race with parallel SS runs
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    Y_norm = normalize_params(Y)

    n = len(X)
    idx = np.random.permutation(n)
    test_size = int(n * 0.1)
    val_size = int(n * 0.1)
    test_idx = idx[:test_size]
    val_idx = idx[test_size:test_size + val_size]
    train_idx = idx[test_size + val_size:]

    X_train = torch.tensor(X[train_idx], device=device, dtype=torch.float32)
    Y_train = torch.tensor(Y_norm[train_idx], device=device, dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], device=device, dtype=torch.float32)
    Y_val = torch.tensor(Y_norm[val_idx], device=device, dtype=torch.float32)
    X_test = torch.tensor(X[test_idx], device=device, dtype=torch.float32)
    Y_test = torch.tensor(Y_norm[test_idx], device=device, dtype=torch.float32)

    if N is not None:
        N_train = torch.tensor(N[train_idx], device=device, dtype=torch.float32)
        offsets_rad_t = _OFFSETS_RAD.to(device)
        w1_t = _W1_RAD.to(device)
        tsat_t = _TSAT_VALUES.to(device)
        log_lo_t = torch.tensor(LOG_PARAM_LO, device=device, dtype=torch.float32)
        log_range_t = torch.tensor(LOG_PARAM_HI - LOG_PARAM_LO, device=device, dtype=torch.float32)

    model = DRONE().to(device)
    if device.type == 'cuda' and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')

    # Loss weighted by log-range² so equal normalized MSE → equal percentage error
    log_ranges = torch.tensor(LOG_PARAM_HI - LOG_PARAM_LO, device=device, dtype=torch.float32)
    loss_w = (log_ranges ** 2)
    loss_w = loss_w / loss_w.mean()
    def criterion(pred, target):
        return (loss_w * (pred - target) ** 2).mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, fused=(device.type == 'cuda'))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0/WARMUP_EPOCHS, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - WARMUP_EPOCHS, eta_min=1e-5)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    n_train = len(train_idx)
    n_val   = len(val_idx)
    n_batches_train = n_train // batch_size
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

                if N is not None and CONS_LAMBDA > 0:
                    nb = N_shuf[i * batch_size:(i + 1) * batch_size]
                    pred_log = pred.clamp(0, 1) * log_range_t + log_lo_t
                    pred_params = torch.exp(pred_log)
                    pred_signal = forward_model_torch(
                        pred_params[:,0], pred_params[:,1], pred_params[:,2], pred_params[:,3],
                        nb[:,0], nb[:,1],
                        offsets_rad_t, w1_t, tsat_t, nb[:,2], nb[:,3])
                    cons_loss = ((pred_signal - xb[:, :266]) ** 2).mean()
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
            cosine_scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), _ckpt)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train={train_loss:.6f}, val={val_loss:.6f}")

    model.load_state_dict(torch.load(_ckpt, weights_only=True))
    os.remove(_ckpt)

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
    """Load time-dependent Z-spectra + SelIR for MTF phantoms.
    Ground truth: Newest_fitting_fig2values_1side2sideSelIR_noTIwt_incl11,7T.csv
    (NOadjfac, Rb=1, 1+simul2+SIR fit at 6.5mT).

    Input signal layout (266-dim, matches training):
      [0:24]   = SelIR curve from SS experiment (same phantoms, normalized to last TI=1.0)
      [24]     = last_ti_norm  (0.0 for TI_last=10s,  1.0 for TI_last=13s)
      [25:114] = zeros (89 padding)
      [114:266]= Z-spec at Tsat=[1.0,1.5,2.0,8.0] × 38 offsets (152 features)
    """
    base_td  = Path(__file__).parent / 'experimental_data_time_dependent'
    selir_path = Path(__file__).parent / 'experimental_data' / 'selIR_data_MTF_Gelatin.mat'
    selir_mat  = sio.loadmat(selir_path)

    # SelIR from SS experiment — already normalized to last TI = 1.0
    # MTF10–40: last TI = 13s  →  last_ti_norm = 1.0
    # MTF50+:   last TI = 10s  →  last_ti_norm = 0.0
    selir_data = {
        'MTF10':    (selir_mat['S_MTF10'].flatten().astype(np.float32),    1.0),
        'MTF20':    (selir_mat['S_MTF20'].flatten().astype(np.float32),    1.0),
        'MTF30':    (selir_mat['S_MTF30'].flatten().astype(np.float32),    1.0),
        'MTF40':    (selir_mat['S_MTF40'].flatten().astype(np.float32),    1.0),
        'MTF50pH4': (selir_mat['S_MTF50pH4'].flatten().astype(np.float32), 0.0),
        'MTF100':   (selir_mat['S_MTF100'].flatten().astype(np.float32),   0.0),
    }

    d_mtf100 = sio.loadmat(base_td / 'ZspecTsatVary_MTF100.mat')
    d_mtf50  = sio.loadmat(base_td / 'ZspecTsatVary_MTF50pc_pH4_shorterTsats.mat')
    d_mtf30  = sio.loadmat(base_td / 'ZspecTsatVary_MTF30.mat')
    d_mtf40  = sio.loadmat(base_td / 'ZspecTsatVary_MTF40.mat')
    d_mtf10  = sio.loadmat(base_td / 'Zspec_TsatVary_MTF10_260428.mat')
    d_mtf20  = sio.loadmat(base_td / 'Zspec_TsatVary_MTF20_260428.mat')

    # (Z_mat, tr_schedule, truth, ci)
    samples_cfg = {
        'MTF10':    (d_mtf10['Z_TsatVary_MTF10'],             TR_MTF30_40,
                     {'R1a': 0.418021628, 'M0b': 0.009410272, 'R': 16.31858327, 'T2b': 1.73e-05},
                     {'R1a': (0.40792532,  0.428117937), 'M0b': (0.008436863, 0.010383681),
                      'R': (14.00578214, 18.63138441), 'T2b': (1.68e-05, 1.77e-05)}),
        'MTF20':    (d_mtf20['Z_TsatVary_MTF20'],             TR_MTF30_40,
                     {'R1a': 0.438835733, 'M0b': 0.021880578, 'R': 20.76319057, 'T2b': 1.77e-05},
                     {'R1a': (0.425963561, 0.451707905), 'M0b': (0.020687034, 0.023074121),
                      'R': (18.84126142, 22.68511971), 'T2b': (1.74e-05, 1.80e-05)}),
        'MTF30':    (d_mtf30['Z_TsatVary_MTF30'],             TR_MTF30_40,
                     {'R1a': 0.48569792,  'M0b': 0.037562104, 'R': 26.02784759, 'T2b': 1.75e-05},
                     {'R1a': (0.472049491, 0.499346349), 'M0b': (0.03616498,  0.038959228),
                      'R': (24.05796563, 27.99772955), 'T2b': (1.73e-05, 1.77e-05)}),
        'MTF40':    (d_mtf40['Z_TsatVary_MTF40'],             TR_MTF30_40,
                     {'R1a': 0.481782758, 'M0b': 0.050374598, 'R': 31.05828597, 'T2b': 1.76e-05},
                     {'R1a': (0.464254967, 0.499310548), 'M0b': (0.048385816, 0.052363381),
                      'R': (28.15774789, 33.95882406), 'T2b': (1.74e-05, 1.78e-05)}),
        'MTF50pH4': (d_mtf50['Z_TsatVary_MTF50pc_pH4_2'],    TR_MTF50_100,
                     {'R1a': 0.531545589, 'M0b': 0.073790791, 'R': 29.24496887, 'T2b': 1.80e-05},
                     {'R1a': (0.50622976,  0.556861417), 'M0b': (0.070423474, 0.077158109),
                      'R': (25.68578934, 32.8041484),  'T2b': (1.77e-05, 1.82e-05)}),
        'MTF100':   (d_mtf100['Z_TsatVary_MTF100'],           TR_MTF50_100,
                     {'R1a': 0.6663553,   'M0b': 0.136465418, 'R': 41.0988062,  'T2b': 1.68e-05},
                     {'R1a': (0.626654223, 0.706056378), 'M0b': (0.130645289, 0.142285547),
                      'R': (34.74427252, 47.45333988), 'T2b': (1.66e-05, 1.70e-05)}),
    }

    data = {}
    for name, (Z_mat, tr_sched, truth, ci) in samples_cfg.items():
        # Full Z-spec (7×38=266) — used for diagnostic residual plots only
        Z_exp_2d = Z_mat.T.astype(np.float32)   # (7, 38)
        zspec_flat = Z_exp_2d.flatten()          # (266,) — original full layout

        # Extract Z-spec at Tsat[3:7]: skip first 3 short-Tsat blocks (0.1s,0.2s,0.5s → 3×38=114 features)
        z_long = zspec_flat[3 * _N_OFFS:].copy()   # [114:266] = 152 features
        # TSAT_LONG = TSAT_VALUES[3:] = [1.0, 1.5, 2.0, 8.0] × 38 offsets

        # Build 266-dim hybrid input
        selir_sig, last_ti_n = selir_data[name]
        signal = np.concatenate([
            selir_sig,                                                  # [0:24]   SelIR
            np.array([last_ti_n], dtype=np.float32),                   # [24]     last_ti_norm
            np.zeros(SELIR_ZERO_PAD, dtype=np.float32),                # [25:114] zeros
            z_long,                                                     # [114:266] Z 4 Tsat
        ])  # (266,)

        data[name] = {'signal': signal, 'truth': truth, 'ci': ci, 'tr': tr_sched,
                      'Z_exp_2d': Z_exp_2d}

    return data


def evaluate_experimental(model, device):
    """Evaluate on experimental data vs Newest_fitting_fig2values_1side2sideSelIR_noTIwt_incl11,7T.csv."""
    print("\n" + "="*60)
    print("EXPERIMENTAL VALIDATION (NOadjfac+simul2)")
    print("="*60)

    data = load_experimental_data()

    print(f"\n{'Sample':<10} {'Param':<5} {'Truth':>8} {'Pred':>8} {'Err%':>7}")
    print("-"*45)

    all_errors = []
    per_sample = []
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
# FORWARD MODEL DIAGNOSTIC
# ==================

def validate_against_experiment(run_dir=None):
    """Plot Z_sim - Z_exp residual heatmaps per sample using published ground truth
    + nominal nuisances + correct per-sample TR schedule."""
    data = load_experimental_data()

    nominal_T2a = 0.3
    nominal_Td  = 0.03

    n_samples = len(data)
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for ax, (name, d) in zip(axes, data.items()):
        truth    = d['truth']
        tr_sched = d['tr']

        R1a_v = np.array([truth['R1a']])
        R1b_v = np.ones(1)
        M0b_v = np.array([truth['M0b']])
        R_v   = np.array([truth['R']])
        T2a_v = np.array([nominal_T2a])
        T2b_v = np.array([truth['T2b']])
        Td_v  = np.array([nominal_Td])
        offs  = OFFSETS_HZ[None, :]  # (1, 38)

        Z_sim = compute_zspectrum_batch(R1a_v, R1b_v, M0b_v, R_v, T2a_v, T2b_v, Td_v,
                                         offs, np.array([W1_RAD]), TSAT_VALUES, tr=tr_sched)
        Z_sim_2d = Z_sim.reshape(len(TSAT_VALUES), len(OFFSETS_HZ))

        Z_exp_2d = d['Z_exp_2d']  # (n_tsat, n_offs) stored by load_experimental_data

        residual = Z_sim_2d - Z_exp_2d
        max_abs = np.max(np.abs(residual))

        im = ax.imshow(residual, aspect='auto', cmap='RdBu_r', vmin=-0.15, vmax=0.15,
                       extent=[-0.5, len(OFFSETS_HZ) - 0.5, len(TSAT_VALUES) - 0.5, -0.5])
        ax.set_title(f'{name}: Z_sim − Z_exp  (max|res|={max_abs:.3f})')
        ax.set_xlabel('Offset index')
        ax.set_ylabel('Tsat index')
        ax.set_yticks(range(len(TSAT_VALUES)))
        ax.set_yticklabels([f'{t}s' for t in TSAT_VALUES])
        plt.colorbar(im, ax=ax)

    plt.suptitle('Forward model residuals (published truth + nominal T2a=0.3, Td=0.03)', fontsize=11)
    plt.tight_layout()

    save_path = (Path(run_dir) / 'residual_heatmap.png') if run_dir else Path('residual_heatmap.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Residual heatmaps saved to {save_path}")


# ==================
# RESULT LOGGING
# ==================

def log_run(train_errors, exp_errors, per_sample, train_losses, val_losses, n_samples, n_augment, training_time, description="", ci_data=None):
    """Save loss / comparison plots and a markdown summary to result_history_timedependent/runN/."""
    from datetime import datetime

    base = Path(__file__).parent / 'result_history_timedependent'
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

    if per_sample:  # experimental plots only when experimental data was available
        samples = list(dict.fromkeys(n for n, _, _, _, _ in per_sample))
        params = ['R1a', 'M0b', 'R', 'T2b']
        truth_vals = {p: [] for p in params}
        pred_vals  = {p: [] for p in params}
        for s in samples:
            for p in params:
                t, pr = next((t, pr) for name, param, t, pr, _ in per_sample if name == s and param == p)
                truth_vals[p].append(t)
                pred_vals[p].append(pr)

        y_ranges = {'R1a': (0.0, 1.1), 'M0b': (0.0, 0.22), 'R': (15, 75), 'T2b': (5e-6, 3e-5)}

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
        fig.suptitle('Time-dependent Neural Network Prediction vs Ground Truth', fontsize=13)
        plt.tight_layout()
        plt.savefig(run_dir / 'comparison.png', dpi=150)
        plt.close()

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
        ax.set_ylabel('Absolute Mean Error (%)')
        ax.set_ylim(0, 100)
        ax.set_title('Time-dependent Experimental Validation Error per Sample per Parameter')
        ax.axhline(10, color='red', linestyle='--', linewidth=1.2)
        ax.grid(True, axis='y')
        ax.legend()
        plt.tight_layout()
        plt.savefig(run_dir / 'error_analysis.png', dpi=150)
        plt.close()

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

    if per_sample:
        outliers = {('MTF10', 'T2b')}
        lines.append("## Experimental Validation (NOadjfac+simul2)\n")
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
        all_flat  = [err for _, _, _, _, err in per_sample]
        excl_flat = [err for name, param, _, _, err in per_sample if (name, param) not in outliers]
        lines.append(f"| **Overall** | **{np.mean(all_flat):.1f}%** | **{np.mean(excl_flat):.1f}%** |")

    with open(run_dir / 'result_history.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Results saved to result_history_timedependent/run{run_num}/")
    return run_dir


# Main: simulate -> train -> experimental validation (if data present) -> save results
if __name__ == '__main__':
    validate_against_steady_state()
    X, Y, N = generate_dataset()
    model, device, train_errors, train_losses, val_losses, training_time = train_model(X, Y, N)

    # Experimental validation needs the phantom data, which is not in the public repo.
    root = Path(__file__).parent
    has_data = (root / 'experimental_data_time_dependent').exists() and (root / 'experimental_data').exists()
    if has_data:
        exp_errors, per_sample, ci_data = evaluate_experimental(model, device)
    else:
        print("\nNo experimental data found; skipping experimental validation.")
        exp_errors, per_sample, ci_data = [], [], None

    run_dir = log_run(
        train_errors, exp_errors, per_sample, train_losses, val_losses,
        N_SAMPLES, N_AUGMENT, training_time, ci_data=ci_data,
        description=(
            "DRONE(266→300×2 Tanh→4) hybrid input: [SelIR(24)+last_ti_norm(1)+zeros(89)+Z_4tsat(152)]. "
            f"Tsat={list(TSAT_LONG)}s (long only, no short-Tsat domain gap). "
            "SelIR from SS experiment. T2b range (15e-6, 19e-6). CosineAnnealingLR. "
            "NN-only inference, no optimization. "
            "Ground truth: Newest_fitting_fig2values_1side2sideSelIR_noTIwt_incl11,7T.csv (NOadjfac+simul2)."
        )
    )
    if has_data:
        validate_against_experiment(run_dir=run_dir)
