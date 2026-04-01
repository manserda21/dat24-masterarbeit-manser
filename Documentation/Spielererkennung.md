## Methoden zur Spieleridentifikation (Kurzüberblick)

### Methode 1: Direktes YOLO auf Spieler
Bild -> YOLO -> Spielername (z.B. Modric)

**Idee:**
Jeder Spieler ist eine eigene Klasse im YOLO-Modell

**Problem:**
- Sehr viele Trainingsdaten nötig
- Spieler sehen visuell sehr ähnlich aus (gleiche Trikots)
- Unrobust bei schlechter Bildqualität, Distanz oder Verdeckung


---

### Methode 2: YOLO + Spielerklassifikator
Bild -> YOLO -> Crop -> Klassifikator -> Spielername

**Idee:**
YOLO erkennt Spieler (Bounding Boxes), ein zweites Modell erkennt die Identität

**Problem:**
- Weiterhin datenintensiv
- Visuell schwer unterscheidbar (ähnliche Trikots)
- Klassifikation einzelner Spieler bleibt schwierig


---

### Methode 3: YOLO + Team + Nummer + Mapping
Bild -> YOLO -> Crop -> Team -> Nummer -> Spielername

**Idee:**
Problem wird in mehrere einfachere Schritte zerlegt:
1. Team erkennen (z.B. Real Madrid vs Gegner)
2. Trikotnummer erkennen
3. Nummer mit Aufstellung (Lineup) mappen → Spielername

**Vorteil:**
- Deutlich robuster und realistischer
- Weniger Trainingsdaten nötig
- Gut geeignet für Broadcast-Fußball


---

## Fazit
- Methode 1: einfach, aber unrealistisch
- Methode 2: besser, aber weiterhin schwierig
- Methode 3: modular, robust und am sinnvollsten für die Masterarbeit

---