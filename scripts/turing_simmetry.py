#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rd_wallpaper.py — Reaction–Diffusion sandbox (model-agnostic) + pattern seeds + wallpaper-group projection.

Deps: numpy, matplotlib
Optional: numba (acelera)
Usage examples:
  python rd_wallpaper.py --model gray_scott --pattern spots --steps 20000 --out out.png
  python rd_wallpaper.py --model schnakenberg --pattern labyrinth --sym p4m --out out.png
  python rd_wallpaper.py --model brusselator --pattern stripes --sym p2+pm --out out.png
  python rd_wallpaper.py --model custom --custom custom_model.py --pattern noise --out out.png

Custom model file must define:
  - n_species: int
  - diffusion: list/tuple length n_species
  - params: dict of default params (JSON-serializable recommended)
  - def reaction(U, t, params): returns array same shape as U (n_species, H, W)
Optionally:
  - def init_state(N, seed, params): returns U0
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate as scipy_rotate


# ----------------------------
# Optional acceleration (Numba)
# ----------------------------
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False
    def njit(*args, **kwargs):  # type: ignore
        def deco(fn): return fn
        return deco


# ----------------------------
# Core math
# ----------------------------
@njit(cache=True)
def _laplacian_periodic_2d(u: np.ndarray) -> np.ndarray:
    # u: (H, W)
    H, W = u.shape
    out = np.empty_like(u)
    for i in range(H):
        im = (i - 1) % H
        ip = (i + 1) % H
        for j in range(W):
            jm = (j - 1) % W
            jp = (j + 1) % W
            out[i, j] = (u[im, j] + u[ip, j] + u[i, jm] + u[i, jp] - 4.0 * u[i, j])
    return out


def laplacian(U: np.ndarray, dx: float) -> np.ndarray:
    # U: (S, H, W)
    S, H, W = U.shape
    out = np.empty_like(U)
    inv_dx2 = 1.0 / (dx * dx)
    for s in range(S):
        out[s] = _laplacian_periodic_2d(U[s]) * inv_dx2
    return out


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


# ----------------------------
# Models
# ----------------------------
@dataclass
class RDModel:
    name: str
    n_species: int
    diffusion: np.ndarray
    params: Dict[str, float]
    reaction: Callable[[np.ndarray, float, Dict[str, float]], np.ndarray]
    init_state: Optional[Callable[[int, Optional[int], Dict[str, float]], np.ndarray]] = None


def _ensure_shape(F: np.ndarray, U: np.ndarray) -> np.ndarray:
    if F.shape != U.shape:
        raise ValueError(f"reaction() must return shape {U.shape}, got {F.shape}")
    return F


def model_gray_scott() -> RDModel:
    # U,V in [0,1] usually
    params = dict(F=0.035, k=0.060)
    diffusion = np.array([1.0, 0.5], dtype=float)

    def reaction(U: np.ndarray, t: float, p: Dict[str, float]) -> np.ndarray:
        u, v = U[0], U[1]
        F = p["F"]; k = p["k"]
        uvv = u * v * v
        du = -uvv + F * (1.0 - u)
        dv =  uvv - (F + k) * v
        out = np.empty_like(U)
        out[0] = du
        out[1] = dv
        return out

    def init_state(N: int, seed: Optional[int], p: Dict[str, float]) -> np.ndarray:
        g = _rng(seed)
        U0 = np.zeros((2, N, N), dtype=float)
        U0[0] = 1.0
        # pequeña región perturbada
        r = N // 10
        cx = N // 2
        U0[1, cx-r:cx+r, cx-r:cx+r] = 0.25 + 0.05 * g.random((2*r, 2*r))
        U0[0, cx-r:cx+r, cx-r:cx+r] = 0.75 + 0.05 * g.random((2*r, 2*r))
        # ruido leve global
        U0 += 0.002 * g.standard_normal(U0.shape)
        return U0

    return RDModel("gray_scott", 2, diffusion, params, reaction, init_state)


def model_schnakenberg() -> RDModel:
    # u_t = D_u Δu + a - u + u^2 v
    # v_t = D_v Δv + b - u^2 v
    params = dict(a=0.2, b=1.3)
    diffusion = np.array([1.0, 10.0], dtype=float)

    def reaction(U: np.ndarray, t: float, p: Dict[str, float]) -> np.ndarray:
        u, v = U[0], U[1]
        a = p["a"]; b = p["b"]
        u2v = (u*u) * v
        du = a - u + u2v
        dv = b - u2v
        out = np.empty_like(U)
        out[0] = du
        out[1] = dv
        return out

    def init_state(N: int, seed: Optional[int], p: Dict[str, float]) -> np.ndarray:
        # cerca del equilibrio
        a, b = p["a"], p["b"]
        u_star = a + b
        v_star = b / ((a + b) ** 2)
        g = _rng(seed)
        U0 = np.zeros((2, N, N), dtype=float)
        U0[0] = u_star + 0.01 * g.standard_normal((N, N))
        U0[1] = v_star + 0.01 * g.standard_normal((N, N))
        return U0

    return RDModel("schnakenberg", 2, diffusion, params, reaction, init_state)


def model_brusselator() -> RDModel:
    # u_t = D_u Δu + A - (B+1)u + u^2 v
    # v_t = D_v Δv + Bu - u^2 v
    params = dict(A=1.0, B=3.0)
    diffusion = np.array([1.0, 8.0], dtype=float)

    def reaction(U: np.ndarray, t: float, p: Dict[str, float]) -> np.ndarray:
        u, v = U[0], U[1]
        A = p["A"]; B = p["B"]
        u2v = (u*u) * v
        du = A - (B + 1.0) * u + u2v
        dv = B * u - u2v
        out = np.empty_like(U)
        out[0] = du
        out[1] = dv
        return out

    def init_state(N: int, seed: Optional[int], p: Dict[str, float]) -> np.ndarray:
        A, B = p["A"], p["B"]
        u_star = A
        v_star = B / A
        g = _rng(seed)
        U0 = np.zeros((2, N, N), dtype=float)
        U0[0] = u_star + 0.02 * g.standard_normal((N, N))
        U0[1] = v_star + 0.02 * g.standard_normal((N, N))
        return U0

    return RDModel("brusselator", 2, diffusion, params, reaction, init_state)


def model_fitzhugh_nagumo() -> RDModel:
    # u: activator, v: inhibitor
    # u_t = D_u Δu + u - u^3/3 - v + I
    # v_t = D_v Δv + (u + a - b v)/tau
    params = dict(a=0.7, b=0.8, tau=12.5, I=0.5)
    diffusion = np.array([1.0, 2.0], dtype=float)

    def reaction(U: np.ndarray, t: float, p: Dict[str, float]) -> np.ndarray:
        u, v = U[0], U[1]
        a = p["a"]; b = p["b"]; tau = p["tau"]; I = p["I"]
        du = u - (u*u*u)/3.0 - v + I
        dv = (u + a - b*v) / tau
        out = np.empty_like(U)
        out[0] = du
        out[1] = dv
        return out

    def init_state(N: int, seed: Optional[int], p: Dict[str, float]) -> np.ndarray:
        g = _rng(seed)
        U0 = np.zeros((2, N, N), dtype=float)
        U0[0] = -1.0 + 0.2 * g.standard_normal((N, N))
        U0[1] =  1.0 + 0.2 * g.standard_normal((N, N))
        return U0

    return RDModel("fitzhugh_nagumo", 2, diffusion, params, reaction, init_state)


def model_oregonator() -> RDModel:
    # Un Oregonator reducido (3 especies) — útil para patrones oscilatorios
    # Fuente estándar (Field–Körös–Noyes), forma adimensional típica.
    params = dict(q=0.002, f=1.4, eps=0.01)
    diffusion = np.array([1.0, 0.6, 0.3], dtype=float)

    def reaction(U: np.ndarray, t: float, p: Dict[str, float]) -> np.ndarray:
        x, y, z = U[0], U[1], U[2]
        q = p["q"]; f = p["f"]; eps = p["eps"]
        # dx/dt = (1/eps) ( qy - xy + x(1-x) )
        # dy/dt = -qy - xy + f z
        # dz/dt = x - z
        dx = (q*y - x*y + x*(1.0 - x)) / eps
        dy = -q*y - x*y + f*z
        dz = x - z
        out = np.empty_like(U)
        out[0] = dx
        out[1] = dy
        out[2] = dz
        return out

    def init_state(N: int, seed: Optional[int], p: Dict[str, float]) -> np.ndarray:
        g = _rng(seed)
        U0 = np.zeros((3, N, N), dtype=float)
        U0[0] = 0.5 + 0.01 * g.standard_normal((N, N))
        U0[1] = 1.0 + 0.01 * g.standard_normal((N, N))
        U0[2] = 0.5 + 0.01 * g.standard_normal((N, N))
        return U0

    return RDModel("oregonator", 3, diffusion, params, reaction, init_state)


BUILTINS: Dict[str, Callable[[], RDModel]] = {
    "gray_scott": model_gray_scott,
    "schnakenberg": model_schnakenberg,
    "brusselator": model_brusselator,
    "fitzhugh_nagumo": model_fitzhugh_nagumo,
    "oregonator": model_oregonator,
}


# ----------------------------
# Pattern seeds (model-agnostic)
# ----------------------------
def _grid_xy(N: int) -> Tuple[np.ndarray, np.ndarray]:
    # in [0, 2π)
    x = np.linspace(0.0, 2.0 * math.pi, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="xy")
    return X, Y


def add_pattern_seed(U: np.ndarray, pattern: str, seed: Optional[int], amp: float = 0.05, k: float = 6.0) -> np.ndarray:
    """
    Applies a perturbation to the last species (or first if 1 species).
    pattern: noise | stripes | square | hex | labyrinth | spots
    """
    g = _rng(seed)
    S, N, _ = U.shape
    idx = S - 1

    if pattern == "noise":
        U[idx] += amp * g.standard_normal((N, N))
        return U

    X, Y = _grid_xy(N)

    if pattern == "stripes":
        U[idx] += amp * np.cos(k * X)
        return U

    if pattern == "square":
        U[idx] += amp * (np.cos(k * X) + np.cos(k * Y))
        return U

    if pattern == "hex":
        # 3 ondas a 120° (esto favorece hex/honeycomb)
        kx1, ky1 = 1.0, 0.0
        kx2, ky2 = 0.5, math.sqrt(3)/2
        kx3, ky3 = 0.5, -math.sqrt(3)/2
        U[idx] += amp * (
            np.cos(k * (kx1 * X + ky1 * Y)) +
            np.cos(k * (kx2 * X + ky2 * Y)) +
            np.cos(k * (kx3 * X + ky3 * Y))
        )
        return U

    if pattern == "labyrinth":
        # ruido band-limit en Fourier alrededor de |k| ~ k
        noise = g.standard_normal((N, N))
        F = np.fft.fft2(noise)
        fx = np.fft.fftfreq(N) * N
        fy = np.fft.fftfreq(N) * N
        FX, FY = np.meshgrid(fx, fy, indexing="xy")
        KR = np.sqrt(FX*FX + FY*FY)
        band = np.exp(-0.5 * ((KR - k) / (0.35 * k + 1e-6)) ** 2)
        filtered = np.fft.ifft2(F * band).real
        filtered = (filtered - filtered.mean()) / (filtered.std() + 1e-8)
        U[idx] += amp * filtered
        return U

    if pattern == "spots":
        # semillas gaussianas aleatorias
        n_blobs = max(8, N // 24)
        xs = g.integers(0, N, size=n_blobs)
        ys = g.integers(0, N, size=n_blobs)
        Xp, Yp = np.meshgrid(np.arange(N), np.arange(N), indexing="xy")
        sig = max(2.0, N / 64.0)
        blob = np.zeros((N, N), dtype=float)
        for x0, y0 in zip(xs, ys):
            dx = (Xp - x0 + N//2) % N - N//2
            dy = (Yp - y0 + N//2) % N - N//2
            blob += np.exp(-(dx*dx + dy*dy) / (2.0 * sig*sig))
        blob = (blob - blob.mean()) / (blob.std() + 1e-8)
        U[idx] += amp * blob
        return U

    raise ValueError(f"Unknown pattern: {pattern}")


# ----------------------------
# Wallpaper-like symmetry projection (robust on square pixel grids)
# ----------------------------
def _roll_half(A: np.ndarray, axis: int) -> np.ndarray:
    n = A.shape[axis]
    if n % 2 != 0:
        raise ValueError("Glide/half-shifts require even N.")
    return np.roll(A, shift=n // 2, axis=axis)


def op_I(A: np.ndarray) -> np.ndarray:
    return A


def op_R90(A: np.ndarray) -> np.ndarray:
    return np.rot90(A, 1)


def op_R180(A: np.ndarray) -> np.ndarray:
    return np.rot90(A, 2)


def op_R270(A: np.ndarray) -> np.ndarray:
    return np.rot90(A, 3)


def op_flipx(A: np.ndarray) -> np.ndarray:
    # mirror vertical axis: left-right
    return np.fliplr(A)


def op_flipy(A: np.ndarray) -> np.ndarray:
    # mirror horizontal axis: up-down
    return np.flipud(A)


def op_transpose(A: np.ndarray) -> np.ndarray:
    return A.T


def op_antitranspose(A: np.ndarray) -> np.ndarray:
    # reflect over anti-diagonal: rot90 + transpose (equiv)
    return np.rot90(A.T, 2)


def op_glide_x(A: np.ndarray) -> np.ndarray:
    # glide reflection along x: mirror + half-translation (x axis)
    return _roll_half(op_flipx(A), axis=1)


def op_glide_y(A: np.ndarray) -> np.ndarray:
    return _roll_half(op_flipy(A), axis=0)


def op_glide_diag(A: np.ndarray) -> np.ndarray:
    # rough "diagonal glide" used in p4g-ish constructions
    B = op_transpose(A)
    return _roll_half(B, axis=1)


# ----------------------------
# Hexagonal rotations (approximate on square grid via interpolation)
# ----------------------------
def _rotate_angle(A: np.ndarray, angle: float) -> np.ndarray:
    """Rotate array by angle degrees using interpolation, preserving shape."""
    return scipy_rotate(A, angle, reshape=False, order=1, mode='wrap')


def op_R60(A: np.ndarray) -> np.ndarray:
    return _rotate_angle(A, 60)


def op_R120(A: np.ndarray) -> np.ndarray:
    return _rotate_angle(A, 120)


def op_R240(A: np.ndarray) -> np.ndarray:
    return _rotate_angle(A, 240)


def op_R300(A: np.ndarray) -> np.ndarray:
    return _rotate_angle(A, 300)


def op_mirror_30(A: np.ndarray) -> np.ndarray:
    """Mirror at 30 degrees from horizontal."""
    # Rotate to align axis, flip, rotate back
    rotated = _rotate_angle(A, -30)
    flipped = np.flipud(rotated)
    return _rotate_angle(flipped, 30)


def op_mirror_60(A: np.ndarray) -> np.ndarray:
    """Mirror at 60 degrees from horizontal."""
    rotated = _rotate_angle(A, -60)
    flipped = np.flipud(rotated)
    return _rotate_angle(flipped, 60)


def op_mirror_120(A: np.ndarray) -> np.ndarray:
    """Mirror at 120 degrees from horizontal."""
    rotated = _rotate_angle(A, -120)
    flipped = np.flipud(rotated)
    return _rotate_angle(flipped, 120)


def group_ops(group: str) -> List[Callable[[np.ndarray], np.ndarray]]:
    """
    All 17 wallpaper groups implemented:
    - Oblique: p1, p2
    - Rectangular: pm, pg, pmm, pmg, pgg, cm, cmm
    - Square: p4, p4m, p4g
    - Hexagonal: p3, p3m1, p31m, p6, p6m
    """
    g = group.lower().strip()

    if g == "p1":
        return [op_I]

    if g == "p2":
        return [op_I, op_R180]

    if g == "pm":
        return [op_I, op_flipx]

    if g == "pg":
        return [op_I, op_glide_x]

    if g == "pmm":
        return [op_I, op_flipx, op_flipy, op_R180]

    if g == "pmg":
        # mirrors + one glide family (approx)
        return [op_I, op_flipx, op_glide_y, lambda A: op_R180(op_glide_y(A))]

    if g == "pgg":
        return [op_I, op_glide_x, op_glide_y, lambda A: op_R180(A)]

    if g == "cmm":
        # like pmm but with a "centering" half-shift: average both cells
        return [op_I, op_flipx, op_flipy, op_R180,
                lambda A: _roll_half(A, axis=1),
                lambda A: _roll_half(op_flipx(A), axis=1),
                lambda A: _roll_half(op_flipy(A), axis=1),
                lambda A: _roll_half(op_R180(A), axis=1)]

    if g == "p4":
        return [op_I, op_R90, op_R180, op_R270]

    if g == "p4m":
        return [op_I, op_R90, op_R180, op_R270,
                op_flipx, op_flipy, op_transpose, op_antitranspose]

    if g == "p4g":
        # aproximación práctica en grilla: rotaciones + glides diagonales
        return [op_I, op_R90, op_R180, op_R270,
                op_glide_x, op_glide_y, op_glide_diag, lambda A: op_R90(op_glide_diag(A))]

    # --- Centered rectangular ---
    if g == "cm":
        # reflection + centering (half-shift along one axis)
        return [op_I, op_flipx,
                lambda A: _roll_half(A, axis=0),
                lambda A: _roll_half(op_flipx(A), axis=0)]

    # --- Hexagonal groups (approximate on square grid) ---
    if g == "p3":
        # 3-fold rotation (120°)
        return [op_I, op_R120, op_R240]

    if g == "p3m1":
        # 3-fold rotation + mirrors through rotation centers
        return [op_I, op_R120, op_R240,
                op_flipx, op_mirror_60, op_mirror_120]

    if g == "p31m":
        # 3-fold rotation + mirrors NOT through rotation centers
        return [op_I, op_R120, op_R240,
                op_flipy, op_mirror_30,
                lambda A: _rotate_angle(op_flipy(A), 120)]

    if g == "p6":
        # 6-fold rotation (60°)
        return [op_I, op_R60, op_R120, op_R180, op_R240, op_R300]

    if g == "p6m":
        # 6-fold rotation + mirrors
        return [op_I, op_R60, op_R120, op_R180, op_R240, op_R300,
                op_flipx, op_flipy, op_mirror_30, op_mirror_60,
                op_mirror_120,
                lambda A: _rotate_angle(op_flipy(A), 60)]

    raise ValueError(f"Unsupported group '{group}'. Try one of: "
                     f"p1,p2,pm,pg,cm,pmm,pmg,pgg,cmm,p4,p4m,p4g,p3,p3m1,p31m,p6,p6m")


def project_group(A: np.ndarray, group: str) -> np.ndarray:
    ops = group_ops(group)
    acc = np.zeros_like(A, dtype=float)
    for op in ops:
        acc += op(A)
    return acc / float(len(ops))


def project_groups(U: np.ndarray, groups: List[str]) -> np.ndarray:
    if not groups:
        return U
    U2 = U.copy()
    for g in groups:
        for s in range(U2.shape[0]):
            U2[s] = project_group(U2[s], g)
    return U2


# ----------------------------
# Simulation
# ----------------------------
def simulate(
    model: RDModel,
    N: int,
    dx: float,
    dt: float,
    steps: int,
    pattern: str,
    seed: Optional[int],
    pattern_amp: float,
    pattern_k: float,
    sym_groups: List[str],
    sym_every: int,
    clip: Optional[Tuple[float, float]],
    report_every: int,
) -> np.ndarray:
    if model.init_state is None:
        U = _rng(seed).standard_normal((model.n_species, N, N)).astype(float) * 0.01
    else:
        U = model.init_state(N, seed, model.params)

    U = add_pattern_seed(U, pattern=pattern, seed=seed, amp=pattern_amp, k=pattern_k)

    D = model.diffusion.reshape((-1, 1, 1)).astype(float)

    for t in range(steps):
        L = laplacian(U, dx)
        F = _ensure_shape(model.reaction(U, t * dt, model.params), U)
        U = U + dt * (D * L + F)

        if clip is not None:
            U = np.clip(U, clip[0], clip[1])

        if sym_groups and (sym_every > 0) and (t % sym_every == 0) and (t > 0):
            U = project_groups(U, sym_groups)

        if report_every > 0 and (t % report_every == 0):
            # log rápido sin spamear
            mn = float(U.min()); mx = float(U.max()); m = float(U.mean())
            print(f"[{model.name}] step {t:6d}/{steps}  min={mn:+.3f} mean={m:+.3f} max={mx:+.3f}")

    return U


# ----------------------------
# Custom model loader
# ----------------------------
def load_custom_model(path: str) -> RDModel:
    spec = importlib.util.spec_from_file_location("custom_rd_model", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load custom model from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore

    n_species = int(getattr(mod, "n_species"))
    diffusion = np.array(getattr(mod, "diffusion"), dtype=float)
    params = dict(getattr(mod, "params"))
    reaction = getattr(mod, "reaction")

    init_state = getattr(mod, "init_state", None)
    return RDModel("custom", n_species, diffusion, params, reaction, init_state)


# ----------------------------
# All 17 wallpaper groups
# ----------------------------
ALL_17_GROUPS = [
    "p1", "p2", "pm", "pg", "cm",
    "pmm", "pmg", "pgg", "cmm",
    "p4", "p4m", "p4g",
    "p3", "p3m1", "p31m", "p6", "p6m"
]


# ----------------------------
# Multiprocessing batch generation
# ----------------------------
def _generate_single_pattern(args_tuple):
    """Worker function for multiprocessing."""
    (model_name, sym_group, scale, variant_seed, N, steps, 
     output_dir, pattern, pattern_amp, dt, dx, sym_every) = args_tuple
    
    # Adjust pattern_k based on scale (higher scale = larger features)
    pattern_k = 6.0 / scale
    
    model = BUILTINS[model_name]()
    
    try:
        U = simulate(
            model=model,
            N=N,
            dx=dx,
            dt=dt,
            steps=steps,
            pattern=pattern,
            seed=variant_seed,
            pattern_amp=pattern_amp,
            pattern_k=pattern_k,
            sym_groups=[sym_group] if sym_group != "p1" else [],
            sym_every=sym_every,
            clip=None,
            report_every=0,  # Silent
        )
        
        s = model.n_species - 1
        img = U[s]
        
        # Normalize
        vmin = np.percentile(img, 1)
        vmax = np.percentile(img, 99)
        if vmax - vmin < 1e-6:
            vmax = vmin + 1
        img_norm = (img - vmin) / (vmax - vmin)
        
        # Save individual image
        out_path = os.path.join(output_dir, f"{sym_group}_scale{scale}_var{variant_seed}.png")
        plt.figure(figsize=(5, 5), dpi=128)
        plt.axis("off")
        plt.imshow(img_norm, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        plt.tight_layout(pad=0)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        
        return (sym_group, scale, variant_seed, img_norm, out_path)
    
    except Exception as e:
        print(f"Error generating {sym_group} scale={scale} seed={variant_seed}: {e}")
        return None


def generate_gallery(
    output_dir: str,
    model_name: str = "gray_scott",
    groups: Optional[List[str]] = None,
    scales: List[int] = [1, 2, 3],
    variants_per_scale: int = 3,
    N: int = 256,
    steps: int = 15000,
    pattern: str = "spots",
    pattern_amp: float = 0.05,
    dt: float = 0.2,
    dx: float = 1.0,
    sym_every: int = 50,
    n_workers: int = 30,
):
    """Generate a gallery of Turing patterns with all symmetry groups."""
    
    if groups is None:
        groups = ALL_17_GROUPS
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build task list
    tasks = []
    seed_counter = 0
    for sym_group in groups:
        for scale in scales:
            for variant in range(variants_per_scale):
                tasks.append((
                    model_name, sym_group, scale, seed_counter,
                    N, steps, output_dir, pattern, pattern_amp, dt, dx, sym_every
                ))
                seed_counter += 1
    
    print(f"Generating {len(tasks)} patterns using {n_workers} workers...")
    print(f"  Groups: {len(groups)}, Scales: {scales}, Variants/scale: {variants_per_scale}")
    
    # Run in parallel
    with Pool(processes=n_workers) as pool:
        results = pool.map(_generate_single_pattern, tasks)
    
    # Filter successful results
    results = [r for r in results if r is not None]
    print(f"Successfully generated {len(results)} patterns")
    
    # Create combined gallery image
    _create_gallery_grid(results, groups, scales, variants_per_scale, output_dir)
    
    return results


def _create_gallery_grid(results, groups, scales, variants_per_scale, output_dir):
    """Create a grid visualization of all generated patterns."""
    
    # Organize results by group -> scale -> variants
    organized = {}
    for r in results:
        if r is None:
            continue
        sym_group, scale, variant, img, path = r
        if sym_group not in organized:
            organized[sym_group] = {}
        if scale not in organized[sym_group]:
            organized[sym_group][scale] = []
        organized[sym_group][scale].append(img)
    
    n_groups = len(groups)
    n_cols = len(scales) * variants_per_scale
    
    fig, axes = plt.subplots(n_groups, n_cols, figsize=(n_cols * 2, n_groups * 2), dpi=100)
    if n_groups == 1:
        axes = [axes]
    
    for row_idx, group in enumerate(groups):
        col_idx = 0
        for scale in scales:
            variants = organized.get(group, {}).get(scale, [])
            for var_idx in range(variants_per_scale):
                ax = axes[row_idx][col_idx] if n_cols > 1 else axes[row_idx]
                if var_idx < len(variants):
                    ax.imshow(variants[var_idx], cmap="magma", vmin=0, vmax=1)
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(f"s{scale}v{var_idx}", fontsize=8)
                if col_idx == 0:
                    ax.set_ylabel(group, fontsize=10, rotation=0, ha="right", va="center")
                col_idx += 1
    
    plt.tight_layout()
    gallery_path = os.path.join(output_dir, "gallery_all_17.png")
    plt.savefig(gallery_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved gallery: {gallery_path}")


# ----------------------------
# CLI
# ----------------------------
def parse_kv_list(kvs: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in kvs:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got '{item}'")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    
    # Mode selection
    ap.add_argument("--gallery", action="store_true", 
                    help="Generate gallery of all 17 groups with multiple scales and variants")
    
    ap.add_argument("--model", choices=list(BUILTINS.keys()) + ["custom"], default="gray_scott")
    ap.add_argument("--custom", type=str, default=None, help="Path to custom model .py if --model custom")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--dx", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--pattern", choices=["noise", "spots", "stripes", "square", "hex", "labyrinth"], default="spots")
    ap.add_argument("--pattern-amp", type=float, default=0.05)
    ap.add_argument("--pattern-k", type=float, default=6.0)

    ap.add_argument("--sym", type=str, default="none",
                    help="Wallpaper projection groups separated by '+', e.g. p4m or p2+pm. Use 'none' to disable.")
    ap.add_argument("--sym-every", type=int, default=50, help="Apply symmetry projection every N steps.")

    ap.add_argument("--clip", type=str, default="none", help="Clip range like -2,2 or 'none'")
    ap.add_argument("--set", nargs="*", default=[], help="Override params: key=value key=value ...")

    ap.add_argument("--out", type=str, default="out.png")
    ap.add_argument("--species", type=int, default=-1, help="Which species to render (default last).")
    ap.add_argument("--report-every", type=int, default=1000)
    
    # Gallery mode options
    ap.add_argument("--gallery-dir", type=str, default="turing_gallery",
                    help="Output directory for gallery mode")
    ap.add_argument("--scales", type=str, default="1,2,3",
                    help="Comma-separated list of scales for gallery mode")
    ap.add_argument("--variants", type=int, default=3,
                    help="Number of variants per scale in gallery mode")
    ap.add_argument("--workers", type=int, default=30,
                    help="Number of parallel workers for gallery mode")
    ap.add_argument("--gallery-steps", type=int, default=15000,
                    help="Steps per pattern in gallery mode")

    args = ap.parse_args()
    
    # Gallery mode
    if args.gallery:
        scales = [int(s.strip()) for s in args.scales.split(",")]
        generate_gallery(
            output_dir=args.gallery_dir,
            model_name=args.model,
            groups=ALL_17_GROUPS,
            scales=scales,
            variants_per_scale=args.variants,
            N=args.N,
            steps=args.gallery_steps,
            pattern=args.pattern,
            pattern_amp=args.pattern_amp,
            dt=args.dt,
            dx=args.dx,
            sym_every=args.sym_every,
            n_workers=args.workers,
        )
        return

    if args.model == "custom":
        if not args.custom:
            raise SystemExit("Need --custom path/to/model.py when --model custom")
        model = load_custom_model(args.custom)
    else:
        model = BUILTINS[args.model]()

    # override params
    overrides = parse_kv_list(args.set)
    model.params.update(overrides)

    # symmetry
    sym_groups: List[str] = []
    if args.sym.lower().strip() != "none":
        sym_groups = [s.strip() for s in args.sym.split("+") if s.strip()]
        # validate now (fail fast)
        for g in sym_groups:
            _ = group_ops(g)

    # clip
    clip = None
    if args.clip.lower().strip() != "none":
        parts = [p.strip() for p in args.clip.split(",")]
        if len(parts) != 2:
            raise SystemExit("--clip must be 'min,max' or 'none'")
        clip = (float(parts[0]), float(parts[1]))

    U = simulate(
        model=model,
        N=args.N,
        dx=args.dx,
        dt=args.dt,
        steps=args.steps,
        pattern=args.pattern,
        seed=args.seed,
        pattern_amp=args.pattern_amp,
        pattern_k=args.pattern_k,
        sym_groups=sym_groups,
        sym_every=args.sym_every,
        clip=clip,
        report_every=args.report_every,
    )

    s = args.species if args.species >= 0 else (model.n_species - 1)
    img = U[s]

    # Normalize for visualization
    vmin = np.percentile(img, 1)
    vmax = np.percentile(img, 99)
    plt.figure(figsize=(6, 6), dpi=160)
    plt.axis("off")
    plt.imshow(img, cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
    plt.tight_layout(pad=0)

    plt.savefig(args.out, bbox_inches="tight", pad_inches=0)
    print(f"Saved {args.out}")
    print(f"Numba enabled: {_HAVE_NUMBA}")
    if sym_groups:
        print(f"Symmetry projection: {' + '.join(sym_groups)} (every {args.sym_every} steps)")


if __name__ == "__main__":
    main()
