# 🗺️ Plan de trabajo: FootballTrack

Plan para construir FootballTrack **aprendiendo paso a paso**. No se trata de llegar rápido al resultado, sino de entender cada pieza del pipeline:

```text
Video → YOLOv8 (detección) → ByteTrack (IDs) → Color de camiseta (equipos) → Stats → Video anotado
```

---

## 🤝 Cómo trabajamos

Cada tarea lleva una etiqueta que dice quién escribe el código:

| Etiqueta | Significa |
|---|---|
| 🧑 **Tú** | Lo escribes tú. Te doy el concepto y pistas, y después reviso tu código. |
| 🤝 **Juntos** | Tú armas la estructura y yo te ayudo con las partes difíciles. |
| 🤖 **Yo + explicación** | Lo escribo yo porque es algo nuevo o complejo, pero te lo explico línea por línea. Después lo modificas tú para comprobar que lo entendiste. |

**El ciclo de cada tarea:**
1. **Concepto**: qué problema resolvemos y por qué de esta forma.
2. **Código**: según la etiqueta.
3. **Verificación**: lo ejecutas y miras el resultado (imagen, video o números).
4. **Checkpoint**: al cerrar cada fase hay preguntas para comprobar que lo entendiste, y luego se hace commit.

**Organización del código:**
- `experiments/`: scripts numerados de cada paso (`01_leer_video.py`, `02_detectar_imagen.py`…). Es tu "cuaderno de laboratorio" y se versiona, porque muestra el camino de aprendizaje.
- `src/`: el pipeline final y limpio, que se arma en la Fase 6 con lo aprendido en los experimentos.

**Git:** una rama por fase (`fase-1-video`, `fase-2-deteccion`…) y merge a `main` al cerrarla.

> ⚠️ **Sin GPU:** todo corre en CPU. Por eso trabajamos con clips cortos (10–20 s), resolución de 720p y el modelo `yolov8n` (el más liviano). Para iterar rápido usamos `--max-frames` y procesamos solo un trozo del video.

---

## Fase 0: Preparar el entorno

**Objetivo:** tener el entorno instalado y un clip de prueba listo.

- [ ] **0.1 Entorno virtual** 🧑 Tú
  - Crear el `venv`, activarlo e instalar `requirements.txt`.
  - Verificar con `python -c "import ultralytics, supervision, cv2; print('ok')"`.
  - 📘 Aprendes: qué es un entorno virtual y por qué aísla dependencias.
- [ ] **0.2 Fijar versiones en `requirements.txt`** 🤝 Juntos
  - Pasar de `ultralytics` a `ultralytics==X.Y.Z` (usando `pip freeze`).
  - Agregar `scikit-learn` (para k-means) y `numpy`.
  - 📘 Aprendes: por qué fijar versiones evita que el proyecto se rompa en el futuro.
- [ ] **0.3 Conseguir el clip de prueba** 🧑 Tú
  - Descargar un video con `yt-dlp` (cámara de transmisión, plano amplio, con los dos equipos de colores bien distintos).
  - Recortar 15 s con `ffmpeg -ss 00:01:00 -t 15 -i in.mp4 -vf scale=-2:720 data/videos/clip.mp4`.
  - 📘 Aprendes: el clip importa. Con un plano amplio y estable todo funciona mejor; con repeticiones y primeros planos falla.
- [ ] **0.4 Crear `experiments/`** 🧑 Tú

**✅ Terminada cuando:** el import funciona y tienes `data/videos/clip.mp4` (15 s, 720p).

---

## Fase 1: Leer y escribir video con OpenCV

**Objetivo:** entender que un video es solo una secuencia de imágenes (arrays de numpy).

- [ ] **1.1 Abrir un video e inspeccionarlo** 🧑 Tú: `experiments/01_leer_video.py`
  - `cv2.VideoCapture`: imprimir FPS, ancho, alto y número de frames.
  - Guardar el primer frame como `primer_frame.jpg`.
  - 📘 Aprendes: un frame es un `np.ndarray` de forma `(alto, ancho, 3)` en **BGR**, no RGB.
- [ ] **1.2 Recorrer frame a frame y escribir un video** 🤝 Juntos: `02_copiar_video.py`
  - Bucle `while cap.read()` con `cv2.VideoWriter` y el codec `mp4v`.
  - Dibujar el número de frame con `cv2.putText`.
  - 📘 Aprendes: codecs, FPS de salida y por qué hay que llamar a `release()`.
- [ ] **1.3 La forma "supervision"** 🤖 Yo + explicación
  - Rehacer 1.2 con `sv.VideoInfo`, `sv.get_video_frames_generator` y `sv.VideoSink`.
  - 📘 Aprendes: generadores en Python (`yield`) y context managers (`with`).

**🧠 Checkpoint:** ¿Por qué un video de 720p a 30 fps "pesa" tanto en memoria si lo cargas entero? ¿Qué pasa si el FPS del `VideoWriter` no coincide con el del original?

---

## Fase 2: Detección con YOLOv8

**Objetivo:** detectar personas y balón en cada frame.

- [ ] **2.1 YOLO sobre una imagen** 🤝 Juntos: `03_detectar_imagen.py`
  - Cargar `YOLO("yolov8n.pt")` (se descarga solo; moverlo a `models/`).
  - Correr el modelo sobre `primer_frame.jpg` e inspeccionar `results[0].boxes` (`xyxy`, `conf`, `cls`).
  - 📘 Aprendes: qué es un bounding box, la confianza y las clases de COCO (`0 = person`, `32 = sports ball`).
- [ ] **2.2 Pasar a `sv.Detections` y filtrar** 🤖 Yo + explicación
  - `sv.Detections.from_ultralytics(...)`, filtrar por clase y confianza con máscaras booleanas de numpy.
  - Dibujar con `sv.BoxAnnotator` y guardar la imagen.
  - 📘 Aprendes: indexación booleana en numpy (`detections[detections.class_id == 0]`), que se usa en todo el proyecto.
- [ ] **2.3 Detección en todo el video** 🧑 Tú: `04_detectar_video.py`
  - Juntar 1.3 y 2.2. Medir cuántos frames por segundo procesa tu CPU (`time.perf_counter`).
- [ ] **2.4 El problema del balón** 🤝 Juntos (experimento)
  - El balón es pequeño y YOLO lo pierde seguido. Probar:
    - `imgsz=1280` frente a `640` (más resolución detecta mejor objetos chicos, pero es más lento).
    - Un umbral de confianza más bajo **solo para el balón**.
    - `yolov8s` frente a `yolov8n`.
  - Anotar en un `experiments/NOTAS.md` el % de frames con el balón detectado en cada configuración.
  - 📘 Aprendes: los trade-offs entre velocidad y precisión, y a medir antes de opinar.
- [ ] **2.5 Mejor visualización** 🧑 Tú
  - `sv.EllipseAnnotator` para jugadores (elipse en los pies) y `sv.TriangleAnnotator` para el balón.

**🧠 Checkpoint:** ¿Qué es un falso positivo y un falso negativo en tu video? ¿Por qué el público de la grada aparece como `person` y cómo podrías filtrarlo?

---

## Fase 3: Tracking con ByteTrack

**Objetivo:** que cada jugador mantenga el **mismo ID** a lo largo del video.

- [ ] **3.1 Concepto: detección frente a tracking** 🤖 Explicación (sin código)
  - La detección no tiene memoria: cada frame es independiente.
  - El tracking asocia detecciones entre frames (IoU y predicción de movimiento con filtro de Kalman).
  - Cómo funciona ByteTrack: primero empareja detecciones de confianza alta y luego recupera las de confianza baja.
- [ ] **3.2 Agregar ByteTrack** 🤝 Juntos: `05_tracking.py`
  - `tracker = sv.ByteTrack(...)` y `tracker.update_with_detections(detections)`.
  - Etiquetas `#ID` con `sv.LabelAnnotator`.
  - 📘 Aprendes: por qué el tracker se crea **fuera** del bucle (guarda estado entre frames).
- [ ] **3.3 Separar jugadores y balón** 🤝 Juntos
  - Hacer tracking solo de personas. El balón es un único objeto: guardar su última posición conocida y, si falta en pocos frames, mantenerla.
- [ ] **3.4 Evaluar y ajustar** 🧑 Tú (experimento)
  - Contar a ojo los *ID switches* (un jugador que cambia de número).
  - Ajustar `lost_track_buffer`, `track_activation_threshold` y `frame_rate`, y anotar los resultados en `NOTAS.md`.

**🧠 Checkpoint:** ¿Qué pasa con el ID cuando dos jugadores se cruzan? ¿Y cuando uno sale de cámara y vuelve?

---

## Fase 4: Separar equipos por color de camiseta

**Objetivo:** clasificar a cada jugador en el equipo A o B sin entrenar ningún modelo.

- [ ] **4.1 Recortar jugadores** 🧑 Tú: `06_crops.py`
  - `sv.crop_image(frame, xyxy)` para cada detección; guardar ~50 recortes en una carpeta y mirarlos.
- [ ] **4.2 Aislar la camiseta** 🤖 Yo + explicación
  - Tomar solo la zona del torso (por ejemplo, del 20 % al 50 % del alto y la franja central del ancho).
  - Quitar el césped con una máscara HSV de verdes (`cv2.inRange`).
  - 📘 Aprendes: espacios de color (BGR, HSV y Lab) y por qué HSV separa mejor "qué color" de "cuánta luz".
- [ ] **4.3 Color representativo** 🧑 Tú
  - Media frente a mediana de los píxeles que quedan. Comparar en BGR y en Lab.
  - Visualizar: dibujar un cuadrado con el color obtenido al lado de cada recorte.
- [ ] **4.4 K-means con k = 2** 🤝 Juntos: `07_equipos.py`
  - Juntar colores de los primeros ~30 frames, entrenar `KMeans(n_clusters=2)` y luego `predict` en cada frame.
  - 📘 Aprendes: aprendizaje no supervisado, centroides y por qué conviene "entrenar" con varios frames.
- [ ] **4.5 Estabilizar por ID** 🤖 Yo + explicación
  - Un jugador puede salir del equipo A en un frame y del B en el siguiente. Solución: votación por mayoría por `tracker_id` (`collections.Counter`).
- [ ] **4.6 Árbitros y porteros (opcional)** 🤝 Juntos
  - Detectarlos como *outliers*: demasiado lejos de ambos centroides, se marcan como "otro".
- [ ] **4.7 Colorear por equipo** 🧑 Tú
  - Las elipses y etiquetas toman el color del equipo (`sv.ColorPalette` + `color_lookup`).

**🧠 Checkpoint:** ¿Por qué falla este método si un equipo viste de verde? ¿Qué ventaja tendrían los embeddings (Fase 8) frente al color promedio?

---

## Fase 5: Estadísticas

**Objetivo:** calcular posesión y distancia recorrida.

- [ ] **5.1 Puntos de referencia** 🤝 Juntos
  - Pies del jugador: `detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)`.
  - Centro del balón.
- [ ] **5.2 Posesión aproximada** 🧑 Tú: `08_posesion.py`
  - Por frame, el jugador más cercano al balón (distancia euclídea, `np.linalg.norm`).
  - Umbral: si nadie está a menos de N píxeles, nadie tiene la posesión (se mantiene la anterior).
  - Acumular frames por equipo y calcular el %.
  - 📘 Aprendes: vectorizar con numpy en vez de usar bucles `for`.
- [ ] **5.3 Distancia recorrida (píxeles)** 🤝 Juntos: `09_distancia.py`
  - Un diccionario `{tracker_id: [posiciones]}` y la suma de desplazamientos entre frames.
  - Problema: el *jitter* (la caja "tiembla" aunque el jugador esté quieto). Solución: suavizado con media móvil o ignorar movimientos menores a X px.
  - 📘 Aprendes: por qué los píxeles **no son metros**, y que si la cámara se mueve la medida se distorsiona. Ese es el motivo de la homografía de la Fase 8.
- [ ] **5.4 Mostrar y exportar** 🧑 Tú
  - Panel en el video con la posesión en vivo (`cv2.rectangle` + `putText`).
  - Al terminar, guardar `data/output/stats.json` (o un CSV) con la posesión por equipo y la distancia por jugador.

**🧠 Checkpoint:** ¿Por qué un jugador cerca de la cámara "recorre" más píxeles que uno lejano aunque corran igual?

---

## Fase 6: Pipeline final en `src/`

**Objetivo:** pasar de scripts sueltos a un proyecto ordenado y reutilizable.

- [ ] **6.1 Diseñar módulos** 🤝 Juntos (primero en papel)
  ```text
  src/
  ├── tracker.py      → punto de entrada (CLI + bucle principal)
  ├── detection.py    → cargar modelo, detectar y filtrar
  ├── tracking.py     → ByteTrack y manejo del balón
  ├── teams.py        → extraer color, k-means y votación por ID
  ├── stats.py        → posesión y distancia (funciones puras)
  └── annotate.py     → todo el dibujo
  ```
  - 📘 Aprendes: separación de responsabilidades y cuándo usar una clase (`TeamClassifier`) o una función.
- [ ] **6.2 Migrar el código** 🧑 Tú, con revisión mía
- [ ] **6.3 CLI con `argparse`** 🧑 Tú
  - `--video`, `--output`, `--model`, `--conf`, `--max-frames`, `--imgsz`.
- [ ] **6.4 Tests con `pytest`** 🤝 Juntos
  - Probar las funciones puras de `stats.py` con datos inventados (por ejemplo, el jugador a 3 px del balón tiene la posesión).
  - 📘 Aprendes: por qué separar la lógica de la visualización hace el código testeable.
- [ ] **6.5 Barra de progreso y logging** 🧑 Tú (`tqdm`)

**✅ Terminada cuando:** `python src/tracker.py --video data/videos/clip.mp4` genera el video anotado y `stats.json`.

---

## Fase 7: Pulir y publicar (v0.1)

- [ ] **7.1** Generar un GIF demo (`ffmpeg`) y agregarlo al README.
- [ ] **7.2** Actualizar el roadmap del README (marcar lo hecho) y agregar una sección de **Limitaciones** (píxeles frente a metros, balón perdido, uniformes parecidos).
- [ ] **7.3** Revisar el `.gitignore`, limpiar `experiments/` y crear el tag `v0.1.0`.
- [ ] **7.4** Escribir en `NOTAS.md` un resumen de lo aprendido (te sirve para el portfolio o para entrevistas).

---

## Fase 8 (futuro): Mejoras del README

Cada una es un mini-proyecto propio. Antes de empezar cualquiera la planificamos en detalle.

| Mejora | Requiere | Dificultad |
|---|---|---|
| Homografía y vista de radar | Puntos de la cancha (manuales o con un modelo de keypoints) y `cv2.findHomography` | ⭐⭐⭐ |
| Distancia y velocidad en metros | La homografía | ⭐⭐ |
| Diagrama de Voronoi del control de espacio | La homografía + `scipy.spatial` | ⭐⭐ |
| Equipos con embeddings (SigLIP + UMAP) | `transformers` y más CPU o tiempo | ⭐⭐⭐ |
| Dataset propio (jugador, portero, árbitro, balón) | Etiquetar y hacer fine-tuning de YOLO (idealmente con GPU o Colab) | ⭐⭐⭐⭐ |

---

## 📌 Progreso

| Fase | Estado |
|---|---|
| 0. Entorno | ⬜ |
| 1. Video con OpenCV | ⬜ |
| 2. Detección | ⬜ |
| 3. Tracking | ⬜ |
| 4. Equipos | ⬜ |
| 5. Stats | ⬜ |
| 6. Pipeline en `src/` | ⬜ |
| 7. Publicar v0.1 | ⬜ |
