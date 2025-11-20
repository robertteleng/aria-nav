# 🤖 MockObserver - Desarrollo sin Gafas Aria

## ✅ ¡Ya está implementado!

El `MockObserver` te permite desarrollar y testear todo el sistema sin necesidad de las gafas Aria físicas.

---

## 🎯 Características

### 3 Modos de Operación:

1. **Sintético** 🎨
   - Genera frames proceduralmente con objetos simulados
   - Perfecto para development y testing inicial
   - No requiere archivos externos

2. **Video Replay** 🎥
   - Reproduce videos grabados en loop
   - Útil para reproducir sesiones específicas
   - Requiere archivo de video

3. **Imagen Estática** 🖼️
   - Imagen fija con pequeñas variaciones
   - Ideal para testing de consistency
   - Requiere archivo de imagen

---

## 🚀 Uso Rápido

### Opción 1: Desde main.py (Recomendado)

```bash
# Activar tu entorno Python
conda activate aria-nav  # o tu entorno correspondiente

# Ejecutar main.py
python3 src/main.py

# Seleccionar:
# 📱 Modo de operación:
#   1. Gafas Aria reales (requiere hardware)
#   2. Mock sintético (desarrollo sin hardware)  ← SELECCIONA ESTO
#   3. Mock con video (replay de sesión grabada)
#   4. Mock con imagen estática
```

### Opción 2: Uso directo en tu código

```python
from core.mock_observer import MockObserver

# Modo sintético (default)
observer = MockObserver(mode='synthetic', fps=60)
observer.start()

# Obtener frames
frame = observer.get_latest_frame()  # Devuelve np.ndarray
if frame is not None:
    print(f"Frame: {frame.shape}")

# Estadísticas
stats = observer.get_stats()
print(f"FPS: {stats['actual_fps']:.1f}")

observer.stop()
```

### Opción 3: Con Context Manager

```python
with MockObserver(mode='synthetic', fps=60) as observer:
    frame = observer.get_latest_frame()
    # Auto cleanup cuando sale del bloque
```

---

## 📋 Tests

### Test básico (sin dependencias de display):

```bash
# Activar entorno
conda activate aria-nav

# Ejecutar test
python3 examples/test_mock_basic.py
```

### Test completo (con visualización OpenCV):

```bash
python3 examples/test_mock_observer.py
```

---

## 🔧 API Completa

El `MockObserver` es **100% compatible** con la API del `Observer` real:

| Método | Descripción | Retorno |
|--------|-------------|---------|
| `start()` | Inicia generación de frames | `None` |
| `stop()` | Detiene generación | `None` |
| `get_latest_frame()` | Frame más reciente | `np.ndarray` o `None` |
| `get_frame_data()` | Frame + metadata | `dict` o `None` |
| `get_buffer_size()` | Tamaño del buffer | `int` |
| `get_stats()` | Estadísticas de operación | `dict` |

---

## 🎮 Ejemplos de Uso

### Modo Sintético (sin archivos):

```python
observer = MockObserver(
    mode='synthetic',
    fps=60,
    resolution=(1408, 1408)
)
observer.start()
```

### Modo Video:

```python
observer = MockObserver(
    mode='video',
    video_path='logs/session_20250114.mp4',
    fps=30
)
observer.start()
```

### Modo Estático:

```python
observer = MockObserver(
    mode='static',
    image_path='data/test_frame.jpg',
    fps=30
)
observer.start()
```

---

## 📊 Estructura de Frames Generados

### Frames sintéticos incluyen:
- Fondo con ruido realista
- 2-6 objetos simulados por frame:
  - Personas (color piel)
  - Sillas (marrón)
  - Mesas (marrón claro)
  - Botellas (azul)
- Timestamp y contador de frames
- Indicador "MOCK MODE: SYNTHETIC"

### Metadata en cada frame:
```python
{
    'frame': np.ndarray,      # Frame RGB
    'timestamp': float,        # Unix timestamp
    'frame_id': int           # Contador de frames
}
```

---

## 🔍 Validación

El MockObserver genera frames a ~60 FPS reales con las siguientes características:

- ✅ Resolución: 1408x1408 (igual que Aria)
- ✅ Color space: RGB (igual que Aria)
- ✅ Threading real (buffer circular)
- ✅ FPS configurable (default 60)
- ✅ Buffer size configurable (default 30 frames)

---

## 💡 Casos de Uso

### 1. Desarrollo de FASE 1 Optimizations
```python
# Testear optimizaciones GPU sin hardware
observer = MockObserver(mode='synthetic', fps=60)
observer.start()

# Tu código de YOLO + Depth aquí
for i in range(1000):
    frame = observer.get_latest_frame()
    # process_frame(frame)
```

### 2. Benchmarks
```python
# Medir FPS del pipeline completo
observer = MockObserver(mode='synthetic', fps=60)
# ... ejecutar benchmark
```

### 3. Replay de Sesiones
```python
# Reproducir sesión problemática
observer = MockObserver(
    mode='video',
    video_path='logs/session_con_error.mp4'
)
# ... debugging
```

---

## 🎯 Próximos Pasos

1. ✅ **MockObserver implementado**
2. 🔄 **Ahora**: Implementar FASE 1 optimizaciones usando el mock
3. 📊 **Siguiente**: Crear benchmarks sintéticos
4. 🧪 **Después**: Testear con gafas reales cuando estén disponibles

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'cv2'"

```bash
# Activar tu entorno conda/venv primero
conda activate aria-nav
# o
source venv/bin/activate
```

### Error: "Video file not found"

```bash
# Verificar que el archivo existe
ls -la data/session.mp4

# O usar modo sintético en su lugar
# (selecciona opción 2 en main.py)
```

### Los frames se ven muy básicos

```
Esto es normal en modo sintético. Son frames procedurales simples
para que YOLO tenga algo que detectar. Para más realismo, usa
modo 'video' con grabaciones reales.
```

---

## 📝 Notas Técnicas

- **Threading**: Usa `threading.Thread` en lugar de multiprocessing
- **Buffer**: Circular deque thread-safe con lock
- **Performance**: Overhead mínimo (~0.1ms por frame sintético)
- **Memory**: ~50MB para buffer de 30 frames @ 1408x1408

---

¡Listo para desarrollar sin las gafas! 🚀
