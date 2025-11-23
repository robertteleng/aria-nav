# 🧠 Glosario de Conceptos (Aria Navigation System)

Este documento explica los conceptos técnicos clave del proyecto en lenguaje sencillo. Úsalo para entender qué hace tu código y para defender tus decisiones técnicas en entrevistas o presentaciones.

---

## 1. Arquitectura y Procesos

### 🐍 El GIL (Global Interpreter Lock)
**¿Qué es?**
Imagina que Python es una oficina con un solo bolígrafo. Aunque tengas 8 empleados (núcleos de CPU), solo uno puede escribir a la vez. El resto tiene que esperar a que suelte el bolígrafo. Eso es el GIL.

**¿Por qué nos afecta?**
En visión por computador, necesitamos procesar muchas cosas a la vez. Si usáramos `threading` (hilos), el GIL haría que todo fuera lento porque los hilos se pelearían por el "bolígrafo".

**Nuestra Solución: Multiprocessing**
En lugar de contratar más empleados en la misma oficina, **alquilamos oficinas separadas** (Procesos).
*   Cada proceso tiene su propio Python y su propio bolígrafo.
*   El proceso `Main` captura imágenes.
*   El proceso `Worker` procesa la IA.
*   No se bloquean entre sí.

### 📬 Colas (Queues)
**¿Qué son?**
Como los procesos están en "oficinas separadas", no pueden hablarse directamente. Las `Queues` son el correo interno.
*   El `Main` mete una foto en un sobre y la deja en la bandeja de entrada (`Queue`).
*   El `Worker` va a la bandeja, coge el sobre y lo procesa.
*   **Problema:** Meter y sacar cosas del sobre (serializar/pickle) es lento si la foto es enorme. Aquí es donde entra la **Shared Memory** (ver abajo).

---

## 2. Aceleración por Hardware (GPU)

### 🌊 CUDA Streams
**¿Qué son?**
Imagina la GPU como una autopista de 100 carriles.
*   **Sin Streams:** Mandas los coches (tareas) uno detrás de otro por el carril central. La autopista está vacía y desaprovechada.
*   **Con Streams:** Abres varios carriles. Por el carril 1 mandas el tráfico de YOLO. Por el carril 2 mandas el tráfico de Profundidad.
*   **Resultado:** La GPU trabaja en paralelo real. Mientras unos núcleos calculan distancias, otros detectan personas.

### 📏 TensorRT
**¿Qué es?**
Es un traductor experto.
*   Tu modelo en PyTorch está escrito en un lenguaje "fácil de leer" pero lento de ejecutar.
*   TensorRT coge ese modelo y lo reescribe optimizado para tu tarjeta gráfica específica. Elimina pasos innecesarios y fusiona operaciones.
*   **FP16 (Half Precision):** TensorRT también reduce la precisión de los números (de 32 decimales a 16). Pierdes un 0.1% de precisión pero ganas el doble de velocidad.

---

## 3. Visión Artificial

### 👁️ Inferencia (Inference)
**¿Qué es?**
Es el acto de "mirar y decidir".
*   **Entrenamiento:** Es cuando la IA va a la escuela y aprende qué es un gato (tarda días).
*   **Inferencia:** Es cuando la IA ya graduada ve una foto y dice "eso es un gato" (tarda milisegundos).
*   Tu sistema solo hace inferencia. Los modelos ya vienen entrenados.

### 🗺️ SLAM (Simultaneous Localization And Mapping)
**¿Qué es?**
Es lo que haces cuando entras en una habitación oscura.
1.  **Mapping:** Tanteas las paredes para saber cómo es la habitación.
2.  **Localization:** Usas esa información para saber dónde estás tú dentro de ella.
*   Las gafas Aria tienen cámaras laterales para hacer esto. Nosotros usamos esas cámaras para detectar obstáculos periféricos ("cuidado, viene alguien por la izquierda").

---

## 4. Optimizaciones Futuras

### 🤝 Shared Memory (Memoria Compartida)
**¿Qué es?**
Volviendo a la analogía de la oficina:
*   **Ahora (Queues):** Haces una fotocopia del documento y se la envías por correo al otro edificio.
*   **Shared Memory:** Pones el documento en un tablón de anuncios en el pasillo. Los dos empleados pueden verlo sin moverlo ni copiarlo.
*   **Ventaja:** Elimina el tiempo de copia. Es "Zero-Copy". Fundamental para vídeo 4K o alta velocidad.

### 🧊 Quantization (INT8)
**¿Qué es?**
Reducir la calidad de los números al mínimo aceptable.
*   En lugar de usar números con decimales (`3.14159`), usamos solo enteros (`3`).
*   Las operaciones matemáticas con enteros son muchísimo más rápidas para el procesador.
*   Requiere "calibrar" para saber cómo redondear sin meter la pata.
