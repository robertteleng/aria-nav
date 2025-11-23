# 🚀 Hoja de Ruta: De TFM a Producto & Portfolio de Ingeniería

Este documento define la estrategia para transformar el proyecto **Aria Navigation System** en:
1.  Un **Producto Viable (Startup)** atractivo para financiación.
2.  Un **Portfolio Técnico Definitivo** que demuestre competencia en 4 perfiles de ingeniería clave.

---

## 🏗️ Fase 1: Ingeniería de Software (Arquitectura & Calidad)
**Objetivo:** Demostrar capacidad para construir sistemas robustos, mantenibles y escalables, no solo scripts académicos.

- [ ] **Implementar CI/CD (GitHub Actions)**
    - *Tarea:* Configurar pipeline que ejecute tests y linter en cada `push`.
    - *Valor:* Muestra profesionalidad y automatización.
- [ ] **Testing Riguroso**
    - *Tarea:* Alcanzar >80% de cobertura con `pytest`. Unit tests para lógica core y Mocks para hardware.
    - *Valor:* Garantía de calidad y estabilidad.
- [ ] **Tipado Estático Estricto**
    - *Tarea:* Aplicar `mypy` en modo estricto a todo el codebase.
    - *Valor:* Código autodocumentado y prevención de bugs.
- [ ] **Refactor a Clean Architecture**
    - *Tarea:* Desacoplar totalmente la lógica de negocio (`DecisionEngine`) de la infraestructura (`Aria SDK`, `PyAudio`).
    - *Valor:* Demuestra diseño de software avanzado.

---

## ⚡ Fase 2: Edge Computing & Visión (Rendimiento Extremo)
**Objetivo:** Demostrar dominio del hardware, baja latencia y optimización de recursos.

- [ ] **Implementar Shared Memory (Zero-Copy)**
    - *Tarea:* Reemplazar `multiprocessing.Queue` por `multiprocessing.shared_memory` para el paso de frames.
    - *Valor:* Reducción drástica de latencia y uso de CPU. **(Prioridad Alta)**
- [ ] **Optimización con TensorRT (INT8)**
    - *Tarea:* Migrar modelos YOLO y Depth a TensorRT con cuantización INT8.
    - *Valor:* Máximo FPS con mínimo consumo energético (W).
- [ ] **Profiling de Energía y Recursos**
    - *Tarea:* Crear dashboard de consumo (CPU/GPU/RAM/Watts) en tiempo real.
    - *Valor:* Conciencia de las limitaciones del Edge.
- [ ] **Pipeline Híbrido C++/Python**
    - *Tarea:* Reescribir el nodo crítico de visión en C++ (usando pybind11 o independientemente).
    - *Valor:* "Musculo" técnico en lenguajes de bajo nivel.

---

## 🧠 Fase 3: Machine Learning (Modelos & Datos)
**Objetivo:** Demostrar capacidad para entrenar, evaluar y gestionar el ciclo de vida de modelos (MLOps).

- [ ] **Pipeline de Active Learning**
    - *Tarea:* Sistema que guarda automáticamente imágenes con baja confianza para re-entrenamiento.
    - *Valor:* Muestra un sistema que "aprende" con el uso.
- [ ] **Integración de VLM (Vision Language Model)**
    - *Tarea:* Añadir un "Agente de Consulta" (ej. Moondream/Florence-2) para descripciones semánticas complejas.
    - *Valor:* Salto de "detectar objetos" a "entender escenas". **(Killer Feature)**
- [ ] **Fine-Tuning Específico**
    - *Tarea:* Entrenar YOLO con dataset propio de "Obstáculos Urbanos" (mezclando datasets públicos).
    - *Valor:* Capacidad de adaptar modelos a problemas específicos.
- [ ] **MLOps Dashboard**
    - *Tarea:* Integrar MLflow o Weights & Biases para trackear experimentos.
    - *Valor:* Gestión profesional de IA.

---

## 🩺 Fase 4: Ingeniería Biomédica (Factor Humano)
**Objetivo:** Demostrar enfoque en el paciente, seguridad clínica e interacción humano-máquina.

- [ ] **Audio Espacial HRTF (Binaural)**
    - *Tarea:* Implementar audio 3D real que simule la función de transferencia de la cabeza.
    - *Valor:* Interfaz sensorial basada en neurociencia, no solo "volumen".
- [ ] **Gestión de Carga Cognitiva**
    - *Tarea:* Algoritmo de filtrado de audio basado en estrés/velocidad del usuario (menos avisos si camina rápido/estresado).
    - *Valor:* Diseño centrado en el paciente.
- [ ] **Métricas de Seguridad (Safety KPIs)**
    - *Tarea:* Telemetría de "Time-to-Collision" y "Obstáculos no avisados".
    - *Valor:* Validación clínica de la eficacia del dispositivo.

---

## 🚀 Fase 5: Startup & Producto (Visión de Negocio)
**Objetivo:** Conseguir financiación y validar mercado.

- [ ] **Abstracción de Hardware (Hardware Agnostic)**
    - *Tarea:* Capa `CameraInterface` que permita usar Webcams baratas o móviles, no solo Aria.
    - *Valor:* Escalabilidad y reducción de riesgo de hardware.
- [ ] **Demo "Wow" (Video Pitch)**
    - *Tarea:* Grabar casos de uso complejos (búsqueda de llaves, lectura de carteles).
    - *Valor:* Herramienta de venta para inversores.
- [ ] **Modo "Batería Baja"**
    - *Tarea:* Degradación elegante del servicio (apagar Depth, bajar FPS) para extender autonomía.
    - *Valor:* Pensamiento de producto real.
