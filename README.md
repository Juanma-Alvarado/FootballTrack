# FootballTrack

Detección y tracking de jugadores + balón en un clip de video de fútbol, con stats simples (quién tiene el balón, distancia recorrida), usando YOLOv8 pre-entrenado + ByteTrack (supervision), sin entrenar ningún modelo propio.

## Instalación
pip install -r requirements.txt

## Uso
python src/tracker.py --video data/videos/clip.mp4
