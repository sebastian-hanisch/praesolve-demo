"""Skalierung von  max c·x,  A x (<=|>=|=) b,  x >= 0: Zeilen mit r_i, Spalten mit t_j (x_j = t_j x'_j). A' = diag(r) A diag(t), b' = r b, c' = t c; der Optimalwert bleibt gleich, Lösung x = t x', Duale y = r y'.
Verfahren: Equilibrierung (Max-Norm 1), geometrisch (Wurzel aus Maximum mal Minimum) und beide mit auf Zweierpotenzen gerundeten Faktoren (`pow2`: die Skalierung selbst rechnet exakt)."""

import math
from dataclasses import dataclass

import numpy as np

import pre_scenario as S

METHODS = ("none", "equilibrate", "geometric", "pow2")
METHOD_LABELS = {"none": "keine", "equilibrate": "Equilibrierung (Max-Norm 1)", "geometric": "Geometrisch (√(max·min))", "pow2": "Zweierpotenzen (geometrisch, gerundet)"}


@dataclass
class Scaled:
    inst: object
    r: np.ndarray
    t: np.ndarray
    method: str
    sweeps: int = 0

    def unscale_x(self, x):
        return np.asarray(x, dtype=float) * self.t

    def unscale_y(self, y):
        return np.asarray(y, dtype=float) * self.r


def spread(A):
    """Zehnerpotenzen zwischen dem größten und kleinsten Betrag der Nichtnullen."""
    a = np.abs(np.asarray(A, dtype=float))
    a = a[a > 0]
    return float(math.log10(a.max() / a.min())) if a.size else 0.0


def _factors(A, method, sweeps):
    m, n = A.shape
    r, t = np.ones(m), np.ones(n)
    used = 0
    for used in range(1, sweeps + 1):
        B = np.abs(A) * r[:, None] * t
        old = (r.copy(), t.copy())
        for i in range(m):
            row = B[i][B[i] > 0]
            if row.size:
                r[i] /= row.max() if method == "equilibrate" else math.sqrt(row.max() * row.min())
        B = np.abs(A) * r[:, None] * t
        for j in range(n):
            col = B[:, j][B[:, j] > 0]
            if col.size:
                t[j] /= col.max() if method == "equilibrate" else math.sqrt(col.max() * col.min())
        if np.allclose(old[0], r, rtol=1e-3) and np.allclose(old[1], t, rtol=1e-3):
            break
    return r, t, used


def scale(inst, method="pow2", sweeps=10):
    if method not in METHODS:
        raise ValueError(method)
    A, b, c = inst.arrays()
    if method == "none":
        return Scaled(inst, np.ones(inst.m), np.ones(inst.n), method)
    r, t, used = _factors(A, "equilibrate" if method == "equilibrate" else "geometric", 3 if method == "equilibrate" else sweeps)
    if method == "pow2":
        r, t = 2.0 ** np.round(np.log2(r)), 2.0 ** np.round(np.log2(t))
    As = A * r[:, None] * t
    scaled = S.Instance(tuple(tuple(float(v) for v in row) for row in As), tuple(float(v) for v in (b * r)), tuple(float(v) for v in (c * t)), inst.senses, inst.names, inst.row_names, inst.kind)
    return Scaled(scaled, r, t, method, used)
