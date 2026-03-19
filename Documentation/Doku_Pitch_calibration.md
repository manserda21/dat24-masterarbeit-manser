Im ersten Schritt wurde die Pitch-Kalibrierung anhand eines Beispielspiels aus dem SoccerNet-Datensatz untersucht. Dazu wurde der Calibration-Datensatz (calibration-2023) heruntergeladen und ein konkreter Frame (Spiel Malaga vs. Real Madrid, Minute 02:18) ausgewählt.

Die im Datensatz enthaltenen Annotationen der Spielfeldlinien wurden zunächst visualisiert, um zu überprüfen, welche Linien im Bild sichtbar sind und für die Kalibrierung genutzt werden können. Anschließend wurde ein erstes Homography-Modell implementiert, um Bildpunkte aus der Kameraperspektive auf ein standardisiertes Spielfeld (105 x 68 Meter) zu projizieren.

Dabei wurde festgestellt, dass ausschließlich Linien verwendet werden sollten, die auf der Rasenebene liegen, da nur diese für eine planare Homography geeignet sind. Entsprechend wurden insbesondere Linien des Strafraums und der Seitenlinien genutzt, während Torpfosten und Querlatten ausgeschlossen wurden.

Die resultierende Transformation zeigt, dass der sichtbare Bildausschnitt korrekt auf den entsprechenden Bereich des Spielfelds abgebildet werden kann. Da im gewählten Frame nur der linke Strafraumbereich sichtbar ist, wird auch ausschließlich dieser Bereich im transformierten Spielfeld dargestellt, während der restliche Bereich leer bleibt.

Insgesamt konnte damit ein erster funktionierender Prototyp für die Pitch-Kalibrierung erstellt werden, der als Grundlage für weitere Schritte dient, insbesondere für die Projektion von Spielerpositionen in ein metrisches Spielfeldkoordinatensystem.