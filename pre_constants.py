"""Konstanten der Demo Präsolve, Skalierung und Numerik: Regler-Bereiche, Stufen, feste Instanzen für die Auswertung, Presets."""
M_MIN, M_MAX, DEFAULT_M = 3, 40, 6
N_MIN, N_MAX, DEFAULT_N = 3, 40, 8
DENSITY = 0.5
SEED_MAX = 999999
DEFAULT_SEED = 35
SPREAD_EXPS = (0, 2, 4, 6, 8, 10, 12, 14, 16)        # Einheiten-Spanne 10^k der Zeilen- und Spaltenfaktoren (Faktoren 10^(±k/2))
DEFAULT_SPREAD_I = 2
REDUNDANCIES = (0.0, 0.25, 0.5, 0.75, 1.0)           # Verschmutzungsgrad
DEFAULT_RED_I = 2
TOLS = (1e-6, 1e-8, 1e-9, 1e-10, 1e-12)              # Toleranz des Simplex (Pivots, reduzierte Kosten, Harris-Lockerung)
DEFAULT_TOL_I = 2
PERTURBS = (0.0, 1e-7, 1e-6)
DEFAULT_PERTURB_I = 0
METHOD_SHORT = {"none": "keine", "equilibrate": "Equilibrierung", "geometric": "Geometrisch", "pow2": "Zweierpotenzen"}
STEPS = {1: "1 · Präsolve", 2: "2 · Skalierung", 3: "3 · Toleranzen und Harris", 4: "4 · Entartung und Störung"}
SWEEP_SEEDS = tuple(range(100000, 100005))
NEEDLE_SEEDS = tuple(range(100000, 100010))
STALL_SIZES = (8, 16, 24, 32, 40)
NEEDLE_SPREAD = 8
_BASE = {"kind": "messy", "m": 6, "n": 8, "seed": 35, "spread": 2, "red": 2, "pre": True, "scaling": "none", "ratio": "textbook", "tol": 2, "perturb": 0, "step": 1}
PRESETS = {
    "Lehrbuch schmutzig: Präsolve von Hand": {**_BASE, "kind": "dirty"},
    "Schmutziges LP: Präsolve räumt auf": {**_BASE},
    "Fast alles ist Schmutz": {**_BASE, "red": 4},
    "Präsolve beweist die Unzulässigkeit": {**_BASE, "kind": "infeasible"},
    "Spanne 10^14: ohne Skalierung falsch": {**_BASE, "spread": 7, "pre": False, "step": 2},
    "Zweierpotenzen reparieren es": {**_BASE, "spread": 7, "pre": False, "scaling": "pow2", "step": 2},
    "Mehrotra jammt ohne Skalierung": {**_BASE, "spread": 5, "pre": False, "scaling": "pow2", "step": 2},
    "Nadel: der einfache Test liefert Unzulässiges": {**_BASE, "kind": "needle", "m": 10, "n": 10, "seed": 6, "spread": 4, "pre": False, "tol": 1, "step": 3},
    "Harris hält die Nadel": {**_BASE, "kind": "needle", "m": 10, "n": 10, "seed": 6, "spread": 4, "pre": False, "tol": 1, "ratio": "harris", "step": 3},
    "Harris braucht eine Lockerung": {**_BASE, "kind": "needle", "m": 10, "n": 10, "seed": 6, "spread": 4, "pre": False, "tol": 4, "ratio": "harris", "step": 3},
    "Entartete Familie: Störung beendet die Nullschritte": {**_BASE, "kind": "ties", "m": 24, "n": 24, "pre": False, "perturb": 2, "step": 4},
}
PRESET_HELP = {
    "Lehrbuch schmutzig: Präsolve von Hand": "Sieben Zeilen, zwei Dienste: der Präsolve entfernt 6 Zeilen (1 leere, 1 redundante mit größter Aktivität 10 ≤ 100, 1 doppelte und 3 Singleton-Zeilen, die zu den Schranken x1 ≤ 4, x2 ≤ 6 und x1 ≤ 3 werden). Übrig bleiben die Lagerfläche und zwei Schranken-Zeilen (3 × 2). Optimum (2, 6) mit Wert 36 wie im Lehrbuch, Postsolve-Probe c·x = b·y.",
    "Schmutziges LP: Präsolve räumt auf": "Kern 6 × 8 mit Schmutz (Verschmutzung 50 %) ergibt 17 × 9. Der Präsolve entfernt 1 leere Zeile, 7 Singleton-Zeilen (zu Schranken), 1 redundante und 3 doppelte Zeilen, 1 feste Variable und 1 leere Spalte; das reduzierte LP hat 9 × 7 (mit Schranken-Zeilen). Der Simplex braucht 5 statt 10 Pivots, und c·x = b·y gilt auf dem Original.",
    "Fast alles ist Schmutz": "Verschmutzung 100 %: aus 31 × 10 wird 7 × 5 (17 Singleton-Zeilen, 6 doppelte, 4 redundante und 2 leere Zeilen, 3 feste Variablen, 2 leere Spalten), die Pivots sinken von 13 auf 3.",
    "Präsolve beweist die Unzulässigkeit": "Die Singleton-Zeilen x1 ≤ 4 und x1 ≥ 6 widersprechen sich: der Präsolve beweist die Unzulässigkeit (Dienst 1: untere Schranke 6 liegt über der oberen 4), ohne dass ein Simplex läuft.",
    "Spanne 10^14: ohne Skalierung falsch": "Dasselbe schmutzige 17 × 9 mit Einheiten-Spanne 10^14: der einfache Simplex mit Toleranz 1e-9 liefert eine falsche Lösung (relative Verletzung 0.42, kleinstes Pivotelement relativ 4e-14). Präsolve allein hilft nicht.",
    "Zweierpotenzen reparieren es": "Dieselbe Instanz mit Zweierpotenz-Skalierung: richtig gelöst nach 6 Pivots, kleinstes Pivotelement relativ 0.16, relative Verletzung 7e-17; die Skalierung rechnet exakt und lässt sich ohne Fehler zurückrechnen.",
    "Mehrotra jammt ohne Skalierung": "Einheiten-Spanne 10^10 (17 × 9): Mehrotra aus Stück 8 steht unbearbeitet nach 57 Iterationen still (Verdacht), mit Zweierpotenz-Skalierung sind es 8 Iterationen bis zum Optimum. Der Simplex ist hier noch richtig (10 Pivots).",
    "Nadel: der einfache Test liefert Unzulässiges": "Nadel 10 × 10 (Seed 6) mit Toleranz 1e-8: die Verhältnisse in der Eintrittsspalte sind fast gleich, der einfache Test wählt ein Pivotelement von relativ 1.6e-6, und die Lösung verletzt die Zeilen um 1.6e-2 (relativ).",
    "Harris hält die Nadel": "Dieselbe Nadel mit Harris-Test: größtes Pivotelement 1.0, relative Verletzung 6e-9 (etwa die Toleranz): richtig gelöst.",
    "Harris braucht eine Lockerung": "Toleranz 1e-12 ist kleiner als die Streuung 1e-8 der Verhältnisse: Harris wählt wieder ein kleines Pivotelement (5.7e-4), macht 7 Nullschritte und verletzt die Zeilen um 2.4e-6 (relativ): falsch.",
    "Entartete Familie: Störung beendet die Nullschritte": "Entartete Familie 24 × 24: ohne Störung 8 Nullschritte bei 13 Pivots, mit Störung 1e-6 keiner bei ebenfalls 13 Pivots. Die gestörte Endbasis ist auch für das Original zulässig; der Zielwert ändert sich um etwa die Störung.",
}


def spread_label(i):
    return "sauber" if SPREAD_EXPS[i] == 0 else f"10^{SPREAD_EXPS[i]}"


def red_label(i):
    return f"{REDUNDANCIES[i]:.0%}"


def tol_label(i):
    return f"{TOLS[i]:.0e}".replace("e-0", "e-")


def perturb_label(i):
    return "aus" if PERTURBS[i] == 0 else f"{PERTURBS[i]:.0e}".replace("e-0", "e-")
