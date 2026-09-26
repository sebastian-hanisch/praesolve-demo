"""Instanzen der Demo Präsolve: Auslastungsplanung eines Distributionszentrums als LP  max c·x,  A x (<=|>=|=) b,  x >= 0."""

import random
from dataclasses import dataclass

import numpy as np

KINDS = ("textbook", "dirty", "centre", "degenerate", "random", "mixed", "messy", "needle", "ties", "infeasible", "unbounded")
FIXTURE_KINDS = ("textbook", "dirty", "centre", "degenerate", "infeasible", "unbounded")
KIND_LABELS = {"textbook": "Lehrbuchbeispiel (2 Dienste)", "dirty": "Lehrbuch schmutzig (2 Dienste, 7 Zeilen)", "centre": "Distributionszentrum (5 Dienste, 4 Ressourcen)", "degenerate": "Entartete Ecke (2 Dienste, 4 Ressourcen)", "random": "Zufall (alle Ressourcen begrenzt)",
               "mixed": "Mischung (mit Mindest- und Gleichungs-Bedingungen)", "messy": "Schmutzig (redundant, Schranken, feste Variablen, Einheiten)", "needle": "Nadel (winzige Koeffizienten, alle Zeilen fast bindend)",
               "ties": "Entartete Familie (viele Gleichstände)", "infeasible": "Unzulässig (Widerspruch)", "unbounded": "Unbeschränkt (kein Ende)"}
LE, GE, EQ = "<=", ">=", "="


@dataclass(frozen=True)
class Instance:
    A: tuple                      # m Zeilen mit je n Koeffizienten (Tupel, damit die Instanz hashbar bleibt)
    b: tuple
    c: tuple
    senses: tuple                 # je Zeile "<=", ">=" oder "="
    names: tuple                  # Namen der n Entscheidungsvariablen (Dienste)
    row_names: tuple              # Namen der m Bedingungen (Ressourcen)
    kind: str = "custom"

    @property
    def m(self):
        return len(self.b)

    @property
    def n(self):
        return len(self.c)

    def arrays(self):
        return np.array(self.A, dtype=float).reshape(self.m, self.n), np.array(self.b, dtype=float), np.array(self.c, dtype=float)


def _inst(A, b, c, senses, names, row_names, kind):
    return Instance(tuple(tuple(float(v) for v in row) for row in A), tuple(float(v) for v in b), tuple(float(v) for v in c), tuple(senses), tuple(names), tuple(row_names), kind)


def textbook_instance():
    """Zwei Dienste, drei Ressourcen: max 3 x1 + 5 x2;  x1 <= 4;  2 x2 <= 12;  3 x1 + 2 x2 <= 18. Optimum (2, 6) mit Wert 36, von Hand in zwei Pivots erreichbar."""
    return _inst([[1, 0], [0, 2], [3, 2]], [4, 12, 18], [3, 5], [LE] * 3, ["Express-Pakete", "Palettenversand"], ["Rampenzeit", "Kommissionierstunden", "Lagerfläche"], "textbook")


def dirty_textbook_instance():
    """Das Lehrbuchbeispiel (max 3 x1 + 5 x2; x1 <= 4; 2 x2 <= 12; 3 x1 + 2 x2 <= 18) mit vier Verschmutzungen zum Nachrechnen von Hand: eine engere Singleton-Zeile x1 <= 3, eine doppelte Zeile (2 mal die Lagerfläche),
    eine leere Zeile (0 <= 5) und eine redundante Zeile x1 + x2 <= 100 (die Schranken erlauben höchstens 3 + 6). Optimum (2, 6) mit Wert 36 wie im Lehrbuch."""
    return _inst([[1, 0], [0, 2], [3, 2], [6, 4], [0, 0], [1, 1], [1, 0]], [4, 12, 18, 36, 5, 100, 3], [3, 5], [LE] * 7, ["Express-Pakete", "Palettenversand"],
                 ["Rampenzeit", "Kommissionierstunden", "Lagerfläche", "Lagerfläche (doppelt)", "Leere Zeile", "Gesamtmenge", "Rampenzeit (eng)"], "dirty")


def degenerate_instance():
    """Das Lehrbuchbeispiel mit einer vierten Ressource x1 + x2 <= 8, die durch das Optimum (2, 6) läuft: an der Ecke sind drei Bedingungen zugleich bindend (entartet), die Duale sind nicht eindeutig."""
    return _inst([[1, 0], [0, 2], [3, 2], [1, 1]], [4, 12, 18, 8], [3, 5], [LE] * 4, ["Express-Pakete", "Palettenversand"], ["Rampenzeit", "Kommissionierstunden", "Lagerfläche", "Fahrzeugkapazität"], "degenerate")


def centre_instance():
    """Distributionszentrum mit 5 Diensten und 4 Ressourcen, rückwärts konstruiert: Optimum x = (20, 10, 10, 0, 0) mit den Schattenpreisen y = (5, 0.5, 4, 0) - drei bindende Ressourcen mit sehr verschiedenem Preis
    (Kommissionierstunden 5, Lagerfläche 0.5, Rampenzeit 4) und eine, die nie bindet (Fahrzeugkapazität, Schlupf 160). Sperrgut und Retouren lohnen sich nicht: ihre Deckungsbeiträge liegen unter dem Wert ihres Verbrauchs."""
    return _inst([[2, 1, 3, 1.5, 2.5], [3, 8, 6, 12, 2], [1, 1.5, 2, 2.5, 0.5], [4, 10, 6, 14, 3]], [80, 200, 55, 400], [15.5, 15, 26, 20, 12], [LE] * 4,
                 ["Express-Pakete", "Palettenversand", "Kühlware", "Sperrgut", "Retouren"], ["Kommissionierstunden", "Lagerfläche (m²)", "Rampenzeit (h)", "Fahrzeugkapazität"], "centre")


def infeasible_instance():
    """Widerspruch: x1 <= 4 (Rampenzeit) und x1 >= 6 (Mindestmenge) zugleich."""
    return _inst([[1, 0], [0, 2], [1, 0]], [4, 12, 6], [3, 5], [LE, LE, GE], ["Express-Pakete", "Palettenversand"], ["Rampenzeit", "Kommissionierstunden", "Mindestmenge Express"], "infeasible")


def unbounded_instance():
    """Kein Ende in Sicht: max x1 + x2 unter x1 - x2 <= 2 und -x1 + x2 <= 3 wächst entlang der Richtung (1, 1) ohne Grenze."""
    return _inst([[1, -1], [-1, 1]], [2, 3], [1, 1], [LE, LE], ["Express-Pakete", "Palettenversand"], ["Bedingung 1", "Bedingung 2"], "unbounded")


def _names(m, n):
    return [f"Dienst {j + 1}" for j in range(n)], [f"Ressource {i + 1}" for i in range(m)]


def _coefficients(rng, m, n, density):
    """Nichtnegative Verbrauchskoeffizienten mit Dichte `density`; jede Zeile und Spalte hat mindestens einen Eintrag, Zeile 0 ist dicht (Gesamtkapazität), damit alles beschränkt bleibt."""
    A = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if rng.random() < density:
                A[i][j] = round(rng.uniform(0.5, 5.0), 2)
    for j in range(n):
        A[0][j] = round(rng.uniform(0.5, 5.0), 2)
    for i in range(m):
        if not any(A[i]):
            A[i][rng.randrange(n)] = round(rng.uniform(0.5, 5.0), 2)
    return A


def generate(kind, m, n, density, seed, spread=0, redundancy=0.0):
    """Deterministische Instanz je (kind, m, n, density, seed); Mersenne-Twister mit Zeichenketten-Seed, plattformstabil. spread und redundancy wirken nur bei "messy"."""
    if kind in FIXTURE_KINDS:
        return {"textbook": textbook_instance, "centre": centre_instance, "degenerate": degenerate_instance, "dirty": dirty_textbook_instance, "infeasible": infeasible_instance, "unbounded": unbounded_instance}[kind]()
    if kind == "messy":
        return messy_instance(m, n, seed, spread, redundancy)[0]
    if kind == "needle":
        return needle_instance(m, n, seed, spread)
    if kind == "ties":
        return ties_instance(m, n, seed)
    rng = random.Random(f"pre-{kind}-{m}-{n}-{density}-{seed}")
    names, row_names = _names(m, n)
    c = [round(rng.uniform(1.0, 10.0), 2) for _ in range(n)]
    A = _coefficients(rng, m, n, density)
    if kind == "random":
        b = [round(rng.uniform(40.0, 120.0), 1) for _ in range(m)]
        return _inst(A, b, c, [LE] * m, names, row_names, kind)
    if kind == "mixed":
        x0 = [rng.uniform(1.0, 8.0) for _ in range(n)]
        senses, b = [LE], []
        eq_left = n // 2
        for i in range(1, m):
            r = rng.random()
            sense = LE if r < 0.55 else (GE if r < 0.85 else EQ)
            if sense == EQ:
                eq_left -= 1
                if eq_left < 0:
                    sense = GE
            senses.append(sense)
        for i in range(m):
            act = sum(A[i][j] * x0[j] for j in range(n))
            slack = rng.uniform(0.5, 25.0)
            if senses[i] == EQ:
                b.append(act)                                                                    # nicht runden: sonst können Gleichungen mit gleichem Träger unvereinbar werden
                continue
            b.append(round(act + slack if senses[i] == LE else act - slack, 2))
            if senses[i] == GE and b[-1] < 0:
                b[-1] = 0.0
        return _inst(A, b, c, senses, names, row_names, kind)
    raise ValueError(kind)


def column_scaled(inst, exponent):
    """Schlechte Skalierung: x_j = t_j y_j mit t_j = 10^(exponent * j / (n-1)); Spalten und Deckungsbeiträge werden mit t_j multipliziert. Das Optimum (Wert) bleibt gleich, die Lösung y_j = x_j / t_j
    liegt aber in sehr verschiedenen Größenordnungen: ein Test für die Numerik (Bereich der Einträge 10^exponent)."""
    if exponent == 0:
        return inst
    n = inst.n
    t = [10.0 ** (exponent * j / max(n - 1, 1)) for j in range(n)]
    A = tuple(tuple(a * t[j] for j, a in enumerate(row)) for row in inst.A)
    c = tuple(cj * t[j] for j, cj in enumerate(inst.c))
    return Instance(A, inst.b, c, inst.senses, inst.names, inst.row_names, inst.kind)


def messy_instance(m, n, seed, spread=0, redundancy=0.5):
    """Realistisch schmutziges LP: ein sauberer Zufallskern (m Zeilen, n Dienste, A >= 0) plus protokollierte Verschmutzung (Rückgabe: Instanz, Protokoll):
    doppelte und dominierte Zeilen (skalierte Kopien), Singleton-Zeilen (x_j <= u_j), feste Variablen (x_j <= v und x_j >= v), leere Zeilen, leere Spalten (mit Kosten <= 0 oder mit Oberschranke),
    durch Schranken erzwungen redundante Zeilen und Einheitenfaktoren 10^(+-spread/2) je Zeile und Spalte (Stück/Tonnen). Der Optimalwert des Kerns mit den Schranken bleibt von den Kopien unberührt."""
    rng = random.Random(f"pre-messy-{m}-{n}-{seed}-{redundancy}")                                # ohne spread: derselbe Schmutz, nur die Einheiten ändern sich
    core = generate("random", m, n, 0.5, seed)
    A = [list(r) for r in core.A]
    b = list(core.b)
    c = list(core.c)
    senses = [LE] * m
    xmax = [min(b[i] / A[i][j] for i in range(m) if A[i][j] > 0) for j in range(n)]
    dirt = {"duplicate": 0, "dominated": 0, "singleton": 0, "fixed": 0, "empty_row": 0, "empty_col": 0, "redundant": 0}
    ub_val = {}

    def add_row(coeffs, rhs, sense=LE):
        A.append(coeffs), b.append(rhs), senses.append(sense)

    k_single = round(redundancy * n)
    for j in rng.sample(range(n), min(n, k_single)):
        row = [0.0] * n
        row[j] = 1.0
        ub_val[j] = round(rng.uniform(0.3, 0.9) * xmax[j], 4)
        add_row(row, ub_val[j])
        dirt["singleton"] += 1
    free_cols = [j for j in range(n) if j not in ub_val] or list(range(n))
    fixed = rng.sample(free_cols, min(len(free_cols), max(1 if redundancy > 0 else 0, round(redundancy * n / 3))))
    for j in fixed:
        v = round(0.8 / len(fixed) * xmax[j], 4)                                                 # zusammen höchstens 80 % jeder Ressource: der Kern bleibt zulässig
        lo, hi = [0.0] * n, [0.0] * n
        lo[j] = hi[j] = 1.0
        add_row(lo, v, GE), add_row(hi, v, LE)
        dirt["fixed"] += 1
    for _ in range(round(redundancy * m)):
        i = rng.randrange(m)
        f = round(rng.uniform(0.5, 4.0), 3)
        loose = rng.random() < 0.5
        add_row([f * a for a in A[i]], f * b[i] * (1.15 if loose else 1.0))
        dirt["dominated" if loose else "duplicate"] += 1
    for _ in range(max(1 if redundancy > 0 else 0, round(redundancy * m / 4))):
        add_row([0.0] * n, round(rng.uniform(1.0, 10.0), 2))
        dirt["empty_row"] += 1
    if len(ub_val) >= 2:
        for _ in range(round(redundancy * m / 3)):
            cols = rng.sample(sorted(ub_val), min(len(ub_val), rng.randint(2, 3)))
            row = [0.0] * n
            for j in cols:
                row[j] = round(rng.uniform(0.5, 3.0), 2)
            add_row(row, round(1.5 * sum(row[j] * ub_val[j] for j in cols), 4))
            dirt["redundant"] += 1
    extra = max(0, round(redundancy * n / 4))
    for e in range(extra):
        for row in A:
            row.append(0.0)
        c.append(-1.0 if e % 2 == 0 else round(rng.uniform(1.0, 4.0), 2))
        if e % 2 == 1:
            row = [0.0] * (n + e + 1)
            row[-1] = 1.0
            add_row(row, round(rng.uniform(1.0, 5.0), 2))
        dirt["empty_col"] += 1
    nn = len(c)
    for row in A:
        row.extend([0.0] * (nn - len(row)))
    mm = len(b)
    order = list(range(mm))
    rng.shuffle(order)
    A, b, senses = [A[i] for i in order], [b[i] for i in order], [senses[i] for i in order]
    s = spread / 2.0
    R = [10.0 ** rng.uniform(-s, s) for _ in range(mm)]
    T = [10.0 ** rng.uniform(-s, s) for _ in range(nn)]
    A = [[A[i][j] * R[i] * T[j] for j in range(nn)] for i in range(mm)]
    b = [b[i] * R[i] for i in range(mm)]
    c = [c[j] * T[j] for j in range(nn)]
    names = [f"Dienst {j + 1}" for j in range(nn)]
    row_names = [f"Zeile {i + 1}" for i in range(mm)]
    dirt["core_rows"], dirt["core_cols"] = m, n
    return _inst(A, b, c, senses, names, row_names, "messy"), dirt


def needle_instance(m, n, seed, spread=8):
    """Nadel (Harris-Falle): Zeile 1 ist normal (x_1 + ... <= 1), die Zeilen 2..m haben in Spalte 1 winzige Koeffizienten 10^-p (p in [3, spread]) und rechte Seiten, die um 10^-8 unter dem Verhältnis 1 liegen
    (b_i = a_i1 (1 - 10^-8 u_i)): beim Eintritt von x_1 sind alle Verhältnisse fast gleich, das kleinste gehört zu einer Zeile mit winzigem Pivotelement. Übrige Koeffizienten der Größe 1."""
    rng = random.Random(f"pre-needle-{m}-{n}-{seed}-{spread}")
    A = [[0.0] * n for _ in range(m)]
    b = [1.0] + [0.0] * (m - 1)
    A[0][0] = 1.0
    for j in range(1, n):
        if rng.random() < 0.6:
            A[0][j] = round(rng.uniform(0.5, 5.0), 2)
    for i in range(1, m):
        A[i][0] = 10.0 ** -rng.uniform(3.0, float(max(spread, 3.5)))
        for j in range(1, n):
            if rng.random() < 0.5:
                A[i][j] = round(rng.uniform(0.5, 5.0), 2)
        b[i] = A[i][0] * (1.0 - 1e-8 * rng.uniform(0.5, 2.0))
    for j in range(1, n):
        if not any(A[i][j] for i in range(1, m)):
            A[rng.randrange(1, m)][j] = round(rng.uniform(0.5, 5.0), 2)
    c = [round(rng.uniform(5.0, 10.0), 2)] + [round(rng.uniform(1.0, 3.0), 2) for _ in range(n - 1)]
    names, row_names = _names(m, n)
    return _inst(A, b, c, [LE] * m, names, row_names, "needle")


def ties_instance(m, n, seed):
    """Entartete Familie: 0/1-Koeffizienten (Dichte 0.35), alle rechten Seiten gleich 10: sehr viele Gleichstände im Quotiententest und viele Nullschritte."""
    rng = random.Random(f"pre-ties-{m}-{n}-{seed}")
    A = [[1.0 if rng.random() < 0.35 else 0.0 for _ in range(n)] for _ in range(m)]
    for j in range(n):
        A[rng.randrange(m)][j] = 1.0
    for i in range(m):
        if not any(A[i]):
            A[i][rng.randrange(n)] = 1.0
    c = [round(rng.uniform(1.0, 10.0), 2) for _ in range(n)]
    names, row_names = _names(m, n)
    return _inst(A, [10.0] * m, c, [LE] * m, names, row_names, "ties")
