<div align="center">

# ⚽ FootballTrack

**Detección y tracking de jugadores + balón en video de fútbol**, sin entrenar ningún modelo propio: YOLOv8 pre-entrenado + ByteTrack para seguir a cada jugador y estimar stats simples (posesión, distancia recorrida).

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=flat-square&logo=yolo&logoColor=black)
![Supervision](https://img.shields.io/badge/Supervision-Roboflow-6706CE?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

</div>

<br>

## 📑 Contenido

- [El problema, en una escena](#-el-problema-en-una-escena)
- [La solución](#-la-solución)
- [Qué hace (roadmap)](#-qué-hace-roadmap)
- [Instalación y uso](#-instalación-y-uso)
- [Cómo está construido](#-cómo-está-construido)
- [Recursos útiles](#-recursos-útiles)
- [Licencia](#-licencia)
- [Autor](#-autor)

---

## 🎬 El problema, en una escena

Quería hacer el tutorial de ["Football AI" de Roboflow](https://github.com/roboflow/sports) — detección de jugadores, homografía, clustering de equipos con embeddings, todo el combo. Pero los datasets de Roboflow Universe que usa ese tutorial dejaron de estar disponibles, y reconstruir esa pipeline completa desde cero es un proyecto demasiado ambicioso para empezar.

**FootballTrack** es la versión reducida: mismo espíritu (visión artificial aplicada al fútbol), pero arrancando con un modelo genérico (YOLOv8 entrenado en COCO, que ya detecta personas y balones) en vez de un dataset custom. La idea es construir en capas — cada funcionalidad nueva se agrega sobre esta base, no reinicia el proyecto.

## 💡 La solución

Un pipeline simple sobre un clip de video:

```text
Video → YOLOv8 (detección) → ByteTrack (IDs persistentes) → Color de camiseta (equipos) → Stats
```

Sin API keys, sin dataset propio, sin entrenamiento — todo corre con pesos pre-entrenados y librerías open source.

## ✨ Qué hace (roadmap)

**Fase 1 — Base (en progreso)**
- [ ] Detección de jugadores y balón con YOLOv8n pre-entrenado
- [ ] Tracking con ByteTrack (ID persistente por jugador a lo largo del video)
- [ ] Separación de equipos por color de camiseta (k-means sobre color promedio del bounding box)
- [ ] Stat: jugador más cercano al balón por frame (posesión aproximada)
- [ ] Stat: distancia recorrida por jugador (en píxeles)
- [ ] Video de salida anotado (cajas, IDs, equipo por color)

**Fase 2 — Mejoras (a futuro)**
- [ ] Homografía / vista de radar (proyección a plano cenital de la cancha)
- [ ] Dataset custom de Roboflow (jugador/portero/árbitro/balón) en vez de clases genéricas de COCO
- [ ] Clustering de equipos con embeddings (SigLIP + UMAP) en vez de color, para uniformes parecidos
- [ ] Distancia y velocidad en metros reales (no píxeles), una vez haya homografía
- [ ] Diagrama de Voronoi de control de espacio

## 🔧 Instalación y uso

```bash
git clone https://github.com/Juanma-Alvarado/FootballTrack.git
cd FootballTrack
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/tracker.py --video data/videos/clip.mp4
```

El resultado se guarda en `data/output/`.

## 🏗️ Cómo está construido

| Capa | Tecnologías |
|---|---|
| Detección | Ultralytics YOLOv8 (pesos pre-entrenados en COCO) |
| Tracking | ByteTrack vía `supervision` (Roboflow) |
| Equipos | OpenCV + k-means sobre color de camiseta |
| Video | OpenCV, `yt-dlp` (opcional, para bajar clips de prueba) |

```text
src/            → tracker.py (pipeline principal)
data/videos/    → clips de entrada (no versionados)
data/output/    → videos anotados generados (no versionados)
models/         → pesos descargados (no versionados)
```

## 📚 Recursos útiles

- [Tutorial original que inspiró el proyecto — Football AI (Roboflow)](https://github.com/roboflow/sports)
- [Documentación de Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Documentación de Supervision (Roboflow)](https://supervision.roboflow.com/)

## 📄 Licencia

MIT — ver [LICENSE](LICENSE).

## 👤 Autor

**Juan Manuel Alvarado**
[GitHub](https://github.com/Juanma-Alvarado) · [Portfolio](https://juanma-alvarado.github.io/PortafolioWeb)
