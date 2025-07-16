# POET Project

Dieses Projekt implementiert eine experimentelle Version des **POET-Algorithmus (Paired Open-Ended Trailblazer; https://doi.org/10.48550/arXiv.1901.01753 )**, welcher . Ich habe diesen Algorithmus für die Optimierung in der Spielumgebung **Pong** implementiert. Es werden simultan sowohl Umgebungen als auch neuronale Netzwerke (hier in einer Achitektur die sich im Rahmen des umsetzbaren für den Neuromorphen Chip ODIN halten) zu entwickelt, die zunehmend komplexe Herausforderungen meistern. Ziel ist es, damit sowohl eine generelle Fähigkeit ein Spiel unabhängig von bestimmten Parametern wärend der Lernphase zu entwickeln, als auch durch die Hierarchische Lernstrategie die Möglichkeit immer komplexere Spielsettings zu meistern. 

## Inhalt

- **POET-Varianten**: Verschiedene Implementierungen des POET-Algorithmus, teilweise mit MAP-Elites-Integration.
- **Rekurrente Neuronale Netzwerke**: Agenten, die Pong spielen lernen.
- **Environment Mutation**: Dynamische Generierung neuer Pong-Umgebungen durch Mutation.
- **Visualisierungen**: Plotten von Scores, Environment-Transfers u.v.m.

