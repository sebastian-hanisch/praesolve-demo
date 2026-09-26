# Präsolve, Skalierung und Numerik – was echte Löser vor und beim Lösen tun – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-praesolve-demo.streamlit.app/)**
Neuntes Stück der **Lineare-Programmierung-Reihe** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Kind von [revised-simplex-demo](https://github.com/sebastian-hanisch/revised-simplex-demo). Die Stücke davor haben mehrfach gemessen: bei Spalten über 10¹⁰ liefert der Simplex (absolute Toleranz 10⁻⁹) still falsche Werte, das [Ellipsoid](https://github.com/sebastian-hanisch/ellipsoid-demo) bricht ab, Langschritt und Affine Scaling der [Inneren Punkte](https://github.com/sebastian-hanisch/innere-punkte-demo) stehen still. Echte Modelle sind nicht sauber: doppelte und redundante Zeilen, Schranken als Zeilen, feste Variablen, Einheiten von Stück bis Tonne. Löser bekämpfen das **vor** dem Lösen – **Präsolve** mit **Postsolve** und **Skalierung** – und rechnen im Simplex mit **Toleranzen**, dem **Harris-Quotiententest** und **Störung** gegen Stillstand. Die Demo baut jedes davon und misst, was es bringt. Vier Fragen: **(1) Präsolve** – wie viel schrumpft ein schmutziges LP, und gibt das Postsolve Lösung **und Duale** zurück? **(2) Skalierung** – repariert sie die Ausfälle aus Stück 6 bis 8? **(3) Toleranzen und Harris** – wann liefert der einfache Quotiententest Unzulässiges? **(4) Entartung und Störung** – hilft eine Störung gegen Nullschritte?

**Einordnung in die Reihe:** geplant sind zwölf Stücke, dies ist das neunte (Details in `lp-planung/PLAN.md` des Portfolio-Ordners):

```
Tableau-Simplex (Wurzel)                                                                  [gebaut: tableau-simplex-demo]
 ├─ Pivotregeln & Entartung ─ Simplex im schlimmsten und im typischen Fall (Klee-Minty)   [gebaut: pivotregeln-demo, klee-minty-demo]
 ├─ Revised Simplex ─ Präsolve, Skalierung & Numerik                                     [gebaut: revised-simplex-demo]  →  [DIESES STÜCK]
 ├─ Dualität & Sensitivität ─ Dualer Simplex & Neuoptimierung                            [gebaut: lp-dualitaet-demo, dualer-simplex-demo]
 ├─ Ellipsoid-Methode (Kontrast: polynomial in der Theorie)                              [gebaut: ellipsoid-demo]
 └─ Innere Punkte ─ PDLP (Verfahren erster Ordnung) ─ Crossover & Simplex gegen Innere Punkte gegen PDLP
      [gebaut: innere-punkte-demo]   →   [nicht gebaut]   →   [nicht gebaut]
```

Ergebnis in Kürze: **Aufräumen und Skalieren reparieren, was der Simplex allein still falsch macht – aber Skalierung ist der Hebel bei der Numerik, nicht Harris, und Präsolve allein hilft dagegen nicht.** Auf einem schmutzigen LP (Kern 6 × 8, Verschmutzung 50 %: 17 × 9) entfernt der Präsolve 1 leere, 7 Singleton- (zu Schranken), 1 redundante und 3 doppelte Zeilen, 1 feste Variable und 1 leere Spalte; das reduzierte LP hat 9 × 7 (mit den Schranken als Zeilen), der Simplex braucht 5 statt 10 Pivots, und das **Postsolve-Zertifikat c·x = b·y** gilt auf dem Original. Bei Verschmutzung 100 % wird 31 × 10 zu 7 × 5 (Pivots 12 auf 3 im Median). Bei Einheiten-Spanne 10¹² bis 10¹⁶ löst der Simplex allein nur noch 3 bis 4 von 5 Instanzen richtig, Mehrotra nur 1 bis 2; **geometrische Skalierung und Zweierpotenzen halten den Simplex bei 5 von 5** (Mehrotra bei 4 bis 5), die Equilibrierung mit drei Sweeps nicht. Auf der **Nadel** (Harris-Falle) liefert der einfache Quotiententest bei Toleranz 10⁻⁸ in 7 von 10 Läufen eine unzulässige Lösung (relative Verletzung bis 0.31), Harris in keinem; bei 10⁻¹² ist die Toleranz kleiner als die Streuung der Verhältnisse, und beide scheitern. Eine **Störung** beendet die Nullschritte auf der entarteten Familie (bis zu 12 gegen 0) und spart bei großem n einige, nicht alle Pivots.

| Frage | Ergebnis (Auslastungsplanung als Standard-LP max c·x; Kern m × n = 6 × 8, 5 feste Instanzen Seeds 100000–100004, Nadel und Familie mit 10 bzw. 5 festen Instanzen; Referenz: Zweierpotenz-Skalierung, dann einfacher Simplex; vollständig deterministisch) |
|---|---|
| **Stimmt das Verfahren?** | ✅ Präsolve + Skalierung + Simplex + Postsolve liefert auf über 300 Instanzen (schmutzig mit allen Verschmutzungen und Spannen bis 10¹⁰, Mischung mit ≥ und =, Zufall, Nadel, Familie, Fixtures) den Optimalwert von HiGHS, ein primal und dual zulässiges Paar auf dem Original und **c·x = b·y bis 10⁻⁶ relativ**; unzulässige Instanzen (Widerspruch durch Singleton-Zeilen oder parallele Zeilen) beweist der Präsolve ohne Simplex, unbeschränkte bleiben unbeschränkt. Jede der sechs Regeln hat einen Fall von Hand, einen, in dem sie nicht greifen darf, und die Ketten (eine Reduktion löst die nächste aus) |
| **Lehrbuch schmutzig (von Hand)** | Sieben Zeilen: der Präsolve entfernt 6 (1 leere, 1 redundante: größte Aktivität 10 ≤ 100, 1 doppelte, 3 Singleton-Zeilen zu den Schranken x1 ≤ 4, x2 ≤ 6, x1 ≤ 3), übrig bleiben die Lagerfläche und zwei Schranken-Zeilen (3 × 2). Optimum (2, 6), Wert 36 |
| **Präsolve über den Verschmutzungsgrad** | Verschmutzung 0 / 25 / 50 / 75 / 100 %: Zeilen 6 → 6 / 13 → 8 / 17 → 10 / 24 → 11 / 31 → 9, Spalten 8 → 8 / 8 → 7 / 9 → 7 / 10 → 6 / 10 → 5, Nichtnullen 29 → 29 / 42 → 27 / 53 → 29 / 61 → 26 / 80 → 18, Pivots 5 → 5 / 5 → 3 / 6 → 4 / 10 → 4 / 12 → 3; alle 5 Läufe je Zelle richtig. Der Sauber-Fall bleibt unberührt |
| **Skalierung, Simplex allein** | Spanne 10^k, richtig gelöst (von 5) ohne Skalierung bei k = 0 / 4 / 8 / 10 / 12 / 14 / 16: **5 / 5 / 5 / 5 / 4 / 3 / 3**; Equilibrierung 5 / 5 / 5 / 5 / 4 / 3 / 2; **geometrisch und Zweierpotenzen immer 5**. Die Spanne der Matrix sinkt von 1 / 13 / 25 Zehnerpotenzen (k = 0 / 8 / 16) auf unter 1 (geometrisch) bzw. unter 1.3 (Zweierpotenzen), die Equilibrierung erreicht 7 bzw. 12 |
| **Skalierung, Mehrotra (Stück 8)** | richtig gelöst ohne Skalierung bei k = 0 / 4 / 6 / 8 / 10 / 12 / 14 / 16: **5 / 5 / 3 / 3 / 2 / 1 / 1 / 2**; Equilibrierung 5 / 4 / 4 / 5 / 1 / 1 / 1 / 1; geometrisch 5 / 5 / 5 / 5 / 5 / 4 / 4 / 4; Zweierpotenzen 5 / 5 / 5 / 5 / 5 / 4 / 4 / 5 |
| **Quotiententest über die Toleranz** | Nadel 10 × 10 (Spanne 8, 10 Instanzen), falsche Läufe (von 10) einfach / Harris bei Toleranz 10⁻⁶ / 10⁻⁸ / 10⁻⁹ / 10⁻¹⁰ / 10⁻¹²: **0 / 0**, **7 / 0**, **3 / 0**, **4 / 0**, **2 / 2**. Bei 10⁻⁸ ist das kleinste Pivotelement des einfachen Tests im Median relativ 1e-6 (Harris: 1.0), die größte relative Verletzung 0.31 (Harris: 1e-8) |
| **Störung gegen Stillstand** | Entartete Familie m = n = 8 / 16 / 24 / 32 / 40: Nullschritte ohne Störung **1 / 6 / 11 / 12 / 12**, mit Störung 10⁻⁷ oder 10⁻⁶ **0**; Pivots ohne / mit 10⁻⁶: 4 / 4, 9 / 9, 20 / 18, 29 / 32, 43 / 34. Die gestörte Endbasis ist in allen Läufen auch für das Original zulässig; der Zielwert ändert sich um 2 · 10⁻⁷ bis 2 · 10⁻⁶ (relativ) |

## Vorab-Hypothesen

| Hypothese (vor der Messung) | Ergebnis |
|---|---|
| Präsolve verkleinert schmutzige LPs stark und spart Pivots | **Bestätigt, mit Einschränkung:** 31 × 10 wird 7 × 5 und 12 Pivots werden 3; das reduzierte LP enthält aber die Schranken als Zeilen (dieser Löser kennt keine), die Zeilenzahl sinkt daher weniger als die Zahl entfernter Zeilen |
| Zweierpotenzen reparieren alles bis 10¹⁶ | **Bestätigt für den Simplex** (5 von 5); die geometrische Skalierung ist genauso gut; **nicht für Mehrotra:** bei 10¹² bis 10¹⁶ bleiben 4 bis 5 von 5 |
| Equilibrierung löst vieles, aber nicht alles | **Bestätigt:** mit drei Sweeps bleibt eine Spanne von 12 Zehnerpotenzen bei k = 16; Simplex 2 von 5, Mehrotra 1 von 5 |
| Harris hat kleinere Restunzulässigkeit und größere Pivots, braucht aber mehr Pivots | **Teils bestätigt:** auf der Harris-Falle (Nadel) größte Pivots (1.0) und Verletzung ≈ Toleranz, dabei kaum mehr Pivots (3 bis 4); **gegen schlechte Skalierung hilft Harris nicht** (siehe unten) |
| Die Störung hilft bei großen entarteten Instanzen, kaum bei kleinen | **Teils bestätigt:** die Nullschritte verschwinden bei jeder Größe; die Pivotzahl sinkt bei n = 24 und 40 (20 auf 18, 43 auf 34), steigt bei n = 32 (29 auf 32) |
| Dualer gegen primalen Simplex (Pivotzahl) | **Nicht gemessen:** siehe Grenzen |

## Was die Demo zeigt

1. **Vier Schritte** (Schritt-Slider): **Präsolve** (Zeilen, Spalten, Nichtnullen vorher und nachher, Protokoll aller Reduktionen, erzeugter Schmutz gegen gefundenen, die Probe c·x − b·y auf dem Original; auf Abruf Ausbeute über den Verschmutzungsgrad) → **Skalierung** (Heatmap log₁₀|a_ij| vorher und nachher, Spanne je Verfahren, Simplex und Mehrotra mit und ohne; auf Abruf Skalierung über die Einheiten-Spanne 10⁰ bis 10¹⁶) → **Toleranzen und Harris** (beide Quotiententests auf der Instanz mit Pivot- und Verletzungskennzahlen; auf Abruf über die Toleranz) → **Entartung und Störung** (Nullschritte ohne und mit Störung auf der Instanz; auf Abruf über die Größe).
2. **Instanzen:** Lehrbuch schmutzig (7 × 2), Lehrbuch, Zentrum, entartete Ecke, Schmutzig (regelbar: Kern, Einheiten-Spanne, Verschmutzungsgrad), Zufall, Mischung, Nadel, Entartete Familie, Unzulässig, Unbeschränkt.
3. **Regler:** Präsolve an/aus, Skalierung (keine / Equilibrierung / geometrisch / Zweierpotenzen), Quotiententest (einfach / Harris), Toleranz (10⁻⁶ bis 10⁻¹²), Störung (aus / 10⁻⁷ / 10⁻⁶).
4. **Ergebnis:** jedes Ergebnis wird gegen eine verlässliche Referenz geprüft (richtig oder falsch mit relativer Unzulässigkeit); "ohne alles" (kein Präsolve, keine Skalierung, einfacher Test, Toleranz 10⁻⁹) wird daneben ausgewiesen.

Presets (11): Lehrbuch schmutzig: Präsolve von Hand, Schmutziges LP: Präsolve räumt auf, Fast alles ist Schmutz, Präsolve beweist die Unzulässigkeit, Spanne 10^14: ohne Skalierung falsch, Zweierpotenzen reparieren es, Mehrotra jammt ohne Skalierung, Nadel: der einfache Test liefert Unzulässiges, Harris hält die Nadel, Harris braucht eine Lockerung, Entartete Familie: Störung beendet die Nullschritte.

## Messwerte der Presets

| Preset | Einstellungen | Ergebnis |
|---|---|---|
| **Lehrbuch schmutzig** | 7 × 2, ohne Skalierung | 6 Zeilen entfernt, 3 × 2 übrig, Optimum (2, 6), Wert 36 |
| **Schmutziges LP** | Kern 6 × 8, 50 %, ohne Skalierung | 17 × 9 → 9 × 7, Pivots 10 → 5, c·x = b·y |
| **Fast alles ist Schmutz** | 100 % | 31 × 10 → 7 × 5, Pivots 13 → 3 |
| **Präsolve beweist die Unzulässigkeit** | Fixture | Dienst 1: untere Schranke 6 über der oberen 4 |
| **Spanne 10^14** | ohne Präsolve und Skalierung | einfacher Simplex falsch: relative Verletzung 0.42, kleinstes Pivotelement 4e-14 |
| **Zweierpotenzen** | dieselbe Instanz | richtig, 6 Pivots, kleinstes Pivotelement 0.16, Verletzung 7e-17 |
| **Mehrotra jammt** | Spanne 10^10 | unbearbeitet Stillstand nach 57 Iterationen, skaliert 8 Iterationen; der Simplex ist noch richtig |
| **Nadel** | 10 × 10, Seed 6, Toleranz 1e-8 | einfacher Test: Pivotelement 1.6e-6, Verletzung 1.6e-2; Harris: 1.0 und 6e-9 |
| **Harris braucht eine Lockerung** | Toleranz 1e-12 | Pivotelement 5.7e-4, 7 Nullschritte, Verletzung 2.4e-6: falsch |
| **Entartete Familie** | 24 × 24, Störung 10⁻⁶ | Nullschritte 8 → 0 bei 13 Pivots |

## Modell und Verfahren

- **Instanz** (`pre_scenario.py`): die Auslastungsplanung der Vorgängerstücke plus **`messy_instance`**: ein sauberer Zufallskern und **protokollierter Schmutz** (doppelte und dominierte Zeilen als skalierte Kopien, Singleton-Zeilen, feste Variablen über zwei Singleton-Zeilen, leere Zeilen, leere Spalten, durch die Schranken erzwungen redundante Zeilen, Einheitenfaktoren 10^(±k/2) je Zeile und Spalte); **`needle_instance`** (Harris-Falle: alle Verhältnisse in der Eintrittsspalte um 10⁻⁸ verschieden, das kleinste gehört zu einem winzigen Pivotelement) und **`ties_instance`** (0/1-Koeffizienten, alle rechten Seiten gleich).
- **Präsolve** (`pre_presolve.py`): leere Zeile, Singleton-Zeile → Schranke, redundante Zeile über die Aktivitätsschranken (ein Widerspruch beweist die Unzulässigkeit), doppelte oder parallele Zeile (die strengere bleibt), feste Variable, leere Spalte; Fixpunkt-Schleife mit Protokoll. Das reduzierte LP verschiebt Unterschranken auf 0 und gibt endliche Oberschranken als Zeilen zurück. **Postsolve:** Lösung aus festen Werten und Verschiebung; **Duale:** reduzierte Zeilen wie gelöst, Schranken-Zeilen über die Quelle der Schranke (y_s = −r_j / a_sj mit r_j aus den bereits bestimmten Dualen, in umgekehrter Reihenfolge), sonst 0.
- **Skalierung** (`pre_scaling.py`): Equilibrierung (Max-Norm 1, 3 Sweeps), geometrisch (√(max·min), bis 10 Sweeps), Zweierpotenzen (geometrisch, gerundet: bitgleiche Rückrechnung); x = t·x', y = r·y'.
- **Simplex** (`pre_simplex.py`): dichtes Tableau wie Stück 1–8 mit Toleranz, einfachem oder Harris-Quotiententest (zweistufig: gelockerte Schrittlänge, dann das größte Pivotelement), Störung von b (Lockerung um 10⁻⁷ bzw. 10⁻⁶ relativ, Mersenne-Twister mit Seed) und Kennzahlen (Nullschritte, kleinstes relatives Pivotelement, ob die gestörte Endbasis für das Original zulässig ist).
- **Auswertung** (`pre_evaluation.py`): Pipeline Präsolve → Skalierung → Simplex → Rückrechnung → Postsolve mit **relativer Unzulässigkeit** (Verletzung / (|b_i| + |a_i|·|x|)) und Zertifikatslücke; Sweeps über Verschmutzung, Spanne, Toleranz, Größe.

## Was nicht funktioniert hat / Grenzen

- **Erste Nadel: Harris half nicht.** Meine erste Nadel-Familie skalierte ganze Zeilen mit 10^-u; dort war Harris nicht besser und bei 10⁻⁹ sogar schlechter als der einfache Test (24 gegen 14 Fehlläufe von 30): die Ausfälle kamen von der absoluten Toleranz gegen die Datenwerte, also von der Skalierung, nicht vom Quotiententest. Erst die Harris-Falle (fast gleiche Verhältnisse, sehr verschiedene Pivotelemente in derselben Spalte bei sonst Größe-1-Daten) zeigt den Vorteil.
- **Harris ist kein Ersatz für Skalierung und braucht die passende Toleranz.** Bei 10⁻¹² unterschreitet die Toleranz die Streuung 10⁻⁸ der Verhältnisse: dann scheitert Harris wie der einfache Test.
- **Präsolve allein repariert die Skalierung nicht.** Bei Spanne 10¹⁴ bleibt der Simplex nach dem Präsolve falsch, mit Zweierpotenzen richtig. Im Modell dieser Demo sind Präsolve und Skalierung unabhängig gebaut; die Reihenfolge Präsolve → Skalierung ist die übliche.
- **Schranken werden zu Zeilen.** Der Löser kennt keine Variablenschranken; deshalb tauchen entfernte Singleton-Zeilen als Schranken-Zeilen im reduzierten LP wieder auf (17 × 9 → 9 × 7 statt kleiner). Echte Löser behandeln Schranken direkt.
- **Nur sechs Regeln.** Keine Doubleton-Aggregation, keine dominierten Spalten, kein Probing, kein Präsolve am Dual, keine erzwingenden Zeilen (alle Variablen an einer Schranke; die Duale wären dort aufwendiger), kein Postsolve der Basis. Die Duale sind für die gebauten Regeln durch die Probe c·x = b·y belegt, nicht für weitere.
- **Dualer gegen primalen Simplex nicht gemessen.** Ein erster Versuch (das Dual-LP mit dem primalen Löser lösen) brauchte 3 bis 5 Mal so viele Pivots, weil es eine Phase 1 braucht; das ist kein dualer Simplex von einer dual zulässigen Basis (Stück 6) und wäre irreführend gewesen.
- **Synthetischer Schmutz, keine echten Modelle** (Netlib, MIPLIB); dichtes Tableau ohne LU-Updates, Bound-Flipping oder Devex; die Bland-Notbremse ist eingeschaltet, auf den Instanzen gibt es Nullschritte, aber keine langen Stillstände.
- **Störung ohne Bereinigung.** Die Demo prüft, ob die gestörte Endbasis für das Original zulässig ist (in allen Läufen ja); echte Löser bereinigen mit einigen Pivots.
- **Plattformabhängigkeit.** Die Numerik-Grenzfälle (Ausfälle bei kleinen Toleranzen, ab Spanne 10¹²) können unter Windows und Linux um einzelne Läufe abweichen; die Tests prüfen dort Bänder.

## Verifikation

- `tests/test_algorithm.py`: **Präsolve + Postsolve gegen HiGHS auf über 300 Instanzen** (Wert, Primal- und Dual-Zulässigkeit auf dem Original, c·x = b·y); unzulässig und unbeschränkt bleiben erhalten; **jede Regel von Hand** (leere Zeile, Singleton-Zeilen mit allen Vorzeichen, redundante Zeilen nur wenn die Schranken es erlauben, doppelte Zeilen mit Vorzeichenwechsel und Widerspruch, feste Variable, leere Spalte mit und ohne Schranke), Fixpunkt-Kette, Lehrbuch schmutzig von Hand; der erzeugte Schmutz wird gefunden; **Zweierpotenz-Skalierung bitgleich rückrechenbar**, Optimum und Duale invariant unter allen Skalierungen, Spanne sinkt; beide Quotiententests erreichen auf sauberen Instanzen dasselbe Optimum, die Nadel trennt sie; Störung deterministisch, Basis und Wert bleiben, Nullschritte verschwinden; Sonderfälle (m = 1, alles redundant, alles fest) und jeder Zweig.
- `tests/test_scenario.py`, `test_evaluation.py`, `test_presets.py` (jede Zahl der Hilfetexte), `test_claims.py` (jede Zahl aus README und App über die echten `ev.*`-Funktionen; Numerik-Grenzfälle nur als Bänder), `test_app.py` (Streamlit-AppTest: Voreinstellung, jedes Preset, jeder Schritt für jede Instanz, Skalierung × Quotiententest, Regler-Randwerte, Permalink-Grenzen, bedingte Regler, Berechnungen auf Abruf, Footer).
- Für die Prüfung genügt **pytest**; `scipy` dient nur als Gegenprobe (`requirements-dev.txt`), die App braucht nur numpy, pandas, plotly und streamlit.

## Lokal starten

```bash
python -m venv venv && venv/Scripts/activate  # Windows; Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -W error::SyntaxWarning`.

## Literatur

- Andersen, E. D., & Andersen, K. D. (1995). *Presolving in linear programming.* Mathematical Programming 71, 221–245.
- Gondzio, J. (1997). *Presolve analysis of linear programs prior to applying an interior point method.* INFORMS Journal on Computing 9(1), 73–91.
- Harris, P. M. J. (1973). *Pivot selection methods of the Devex LP code.* Mathematical Programming 5, 1–28.
- Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1989). *A practical anti-cycling procedure for linearly constrained optimization.* Mathematical Programming 45, 437–474.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
