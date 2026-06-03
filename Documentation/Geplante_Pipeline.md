## Geplante Pipeline

1. **YOLO-Detection**
   - Ein YOLO-Modell läuft über das Spiel
   - Es erkennt nur allgemeine Klassen wie:
     - `player`
     - optional `ball`
     - optional `referee`

2. **Tracking**
   - Die erkannten Spieler werden über mehrere Frames hinweg verfolgt
   - Dadurch entstehen **Track-IDs**
   - Ein Track entspricht idealerweise demselben Spieler über die Zeit

3. **Crops extrahieren**
   - Für die Bounding Boxes der Spieler werden aus den Frames kleine Bildausschnitte erzeugt
   - Diese Crops enthalten jeweils nur einen Spieler

4. **Team-/Spieleridentifikation**
   - Die Crops werden nicht direkt nur über YOLO identifiziert
   - Stattdessen kommen zusätzliche Verfahren dazu, z.B.:
     - **Team-Klassifikator** (`Real Madrid` vs. `not Real Madrid`)
     - **Jersey-Number-Erkennung**
     - **ReID** als zusätzliche visuelle Hilfe / Absicherung

5. **Mapping zur realen Spieleridentität**
   - Erkannte Trikotnummer + Team + Aufstellung des Spiels
   - daraus folgt dann der Spielername
   - Beispiel:
     - `Real Madrid` + `19` -> `Luka Modric`

6. **Pitch-Projektion**
   - Die Spielerpositionen aus dem Bild werden mit der Pitch-Kalibrierung auf das Spielfeld projiziert

7. **Trajektorien und Passnetzwerk**
   - Aus den Tracks auf dem Spielfeld werden Bewegungen und Ballübergänge analysiert
   - Am Ende entsteht ein Passnetzwerk mit echten Spielernamen