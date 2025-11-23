# 📜 Historia del Desarrollo: De TFM Académico a Producto Real

Este documento narra la evolución técnica del **Aria Navigation System**, dividida en dos etapas claras: el proyecto académico (TFM) y su posterior evolución hacia un producto de ingeniería de alto rendimiento.

---

## 🎓 Parte 1: El TFM Académico (Semanas 1-10)
**Objetivo:** Validar la viabilidad de un sistema de navegación para invidentes usando gafas Meta Aria.
**Resultado:** Sistema funcional en Python puro (~18 FPS).

### 🟢 Iteración 1-2: La Base (RGB + Audio)
*   **Logro:** Conexión con gafas Aria y detección YOLO básica.
*   **Audio:** Implementación de zonas espaciales (Izquierda/Centro/Derecha).

### 🟠 Iteración 3-5: Enriquecimiento Sensorial
*   **Profundidad:** Integración de *Depth Anything V2*. El rendimiento cae pero ganamos distancia.
*   **Low-Light:** Mejora de imagen para entornos oscuros.
*   **Movimiento:** Uso del IMU para detectar si el usuario camina o está quieto.

### 🔵 Iteración 6-8: Visión 360º y Dashboards
*   **SLAM:** Activación de cámaras laterales para detectar peligros periféricos.
*   **Routing:** Reescritura del sistema de audio para gestionar 3 cámaras a la vez.
*   **Dashboards:** Creación de visualizaciones Web y 3D para depuración.

> **🏁 Hito Académico:** Aquí termina el alcance del TFM. Un prototipo funcional, modular y validado conceptualmente.

---

## 🚀 Parte 2: El Proyecto Profesional (Post-TFM)
**Objetivo:** Transformar el prototipo en un producto viable comercialmente (Startup) y demostrar ingeniería avanzada.
**Plataforma:** Intel NUC 11 Enthusiast (RTX 2060 6GB).
**Resultado:** Sistema optimizado con CUDA/TensorRT (~25 FPS con carga completa).

### ⚡ Fase 9: Ingeniería de Rendimiento (CUDA)
*   **El Salto:** Abandonamos la ejecución en CPU/MPS.
*   **TensorRT:** Compilación de YOLO a FP16 (2.5x más rápido).
*   **ONNX Runtime:** Migración del modelo de profundidad a CUDA (11.7x más rápido).

### 🧬 Fase 10: Arquitectura Híbrida (Estado Actual)
*   **El Problema:** El Multiprocessing clásico dejaba la GPU infrautilizada por la sincronización.
*   **La Solución:** Arquitectura híbrida **Multiprocessing + CUDA Streams**.
    *   Workers aislados para cámaras SLAM.
    *   Paralelismo real (Streams) dentro del proceso principal para Depth+YOLO.
*   **Impacto:** Latencia reducida a **40ms** y FPS estables en **~25 FPS**.

---

## 📊 Comparativa: TFM vs Producto

| Característica | TFM (Final) | Producto (Actual) |
| :--- | :--- | :--- |
| **Enfoque** | Funcionalidad / UX | Rendimiento / Escalabilidad |
| **Hardware** | Laptop (CPU/MPS) | **Intel NUC 11 + RTX 2060** |
| **FPS** | ~18 FPS (inestable) | **~25 FPS (sólido)** |
| **Latencia** | ~125ms | **~40ms** |
| **Tecnología** | Python Puro / PyTorch | **TensorRT / CUDA / C++ (pronto)** |
| **Arquitectura** | Hilos (Threading) | **Híbrida (Multiproc + Streams)** |

---

## 🔮 Parte 3: Industrialización (En Progreso)
**Objetivo:** Eliminar las limitaciones de Python y preparar el hardware final.
**Estado:** Implementación de Shared Memory completada (pendiente de validación en hardware real).

### Hoja de Ruta Inmediata:
1.  **Shared Memory (Zero-Copy):** ✅ Infraestructura implementada (`SharedMemoryRingBuffer`). Pendiente verificar mejora de latencia.
2.  **Integración C++:** Reescribir módulos críticos (como el post-procesado de profundidad) en C++ para latencia <1ms.
3.  **Optimización Energética:** Ajustar perfiles de energía para modo portátil.

> *Para ver el plan detallado de esta fase, consultar `ROADMAP_CAREER.md`.*

*Este documento es un resumen narrativo. Para el registro técnico exacto de cada commit y versión, consultar `CHANGELOG.md`.*
