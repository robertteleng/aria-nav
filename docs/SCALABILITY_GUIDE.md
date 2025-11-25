# 🚀 Guía de Escalabilidad - aria-nav

## 📊 Capacidad Actual (Baseline)

**Hardware:** RTX 2060 (6GB VRAM), CPU 8-core  
**Rendimiento:** 18 FPS, 55ms latencia  
**Uso:**
- GPU: 44% (56% libre)
- VRAM: 1.3GB (79% libre)  
- CPU: 38% (62% libre)

---

## ✅ Nivel 1: Añadir funcionalidades SIN degradación

**Capacidad disponible:** +1GB VRAM, +15% GPU, +20ms latencia

### Funcionalidades recomendadas:

1. **Object Tracking** (SORT)
   - Costo: +50MB VRAM, +3ms, +5% GPU
   - Beneficio: IDs persistentes, trayectorias
   
2. **Pose Estimation** (MoveNet Lightning)
   - Costo: +150MB VRAM, +10ms, +10% GPU
   - Beneficio: Detectar gestos, posturas peligrosas

3. **OCR on-demand** (EasyOCR)
   - Costo: +300MB VRAM, +25ms (solo cuando se activa)
   - Beneficio: Leer señales, carteles, texto

**Total Nivel 1:** VRAM 1.8GB, Latencia 70ms, GPU 60%, **FPS: ~15-16**

---

## ⚠️ Nivel 2: Añadir con optimizaciones

**Requiere:** Reducir resolución a 768x768, optimizar pipelines

### Funcionalidades avanzadas:

1. **Semantic Segmentation** (DeepLabV3+ Mobile)
   - Costo: +400MB VRAM, +20ms, +15% GPU
   - Beneficio: Segmentar aceras, carreteras, obstáculos

2. **Face Recognition** (FaceNet)
   - Costo: +200MB VRAM, +8ms, +8% GPU
   - Beneficio: Reconocer personas conocidas

3. **Audio Source Localization** (beamforming)
   - Costo: +10% CPU
   - Beneficio: Dirección de sonidos

**Total Nivel 2:** VRAM 2.4GB, Latencia 95ms, GPU 85%, CPU 50%, **FPS: ~12-13**

---

## 🔴 Nivel 3: Requiere hardware adicional

**No cabe en RTX 2060 o requiere procesamiento distribuido**

### Opciones:

1. **Offload a cloud** (LLMs, modelos pesados)
   - Costo: API costs, +100-500ms latency
   - Beneficio: Capacidades ilimitadas

2. **Edge TPU secundaria** (Coral USB)
   - Costo: $60 hardware
   - Beneficio: +30 FPS solo para YOLO, libera GPU

3. **Jetson Orin Nano** (segundo dispositivo)
   - Costo: $499
   - Beneficio: SLAM dedicado, path planning

4. **Upgrade a RTX 4060** (12GB VRAM)
   - Costo: $300
   - Beneficio: 2x capacidad, mismas optimizaciones

---

## 🎯 Arquitectura Escalable Recomendada

### Opción A: Modular con feature flags

```python
# config.py
class Features:
    OBJECT_TRACKING = True      # Nivel 1
    POSE_ESTIMATION = False     # Nivel 1 (desactivado por defecto)
    OCR_ON_DEMAND = True        # Nivel 1
    SEMANTIC_SEGMENTATION = False  # Nivel 2
    FACE_RECOGNITION = False    # Nivel 2
    
    # Auto-detect capacity
    @classmethod
    def auto_enable(cls):
        vram_available = get_vram_free()
        if vram_available > 1500:  # 1.5GB free
            cls.POSE_ESTIMATION = True
        if vram_available > 2000:  # 2GB free
            cls.SEMANTIC_SEGMENTATION = True
```

### Opción B: Procesamiento diferido (bajo demanda)

```python
class OnDemandProcessor:
    """Run expensive models only when triggered"""
    
    def __init__(self):
        self.ocr_model = None  # Lazy load
        self.last_ocr_time = 0
        
    def process_if_needed(self, frame, user_request):
        if user_request == "read_text" and time.time() - self.last_ocr_time > 5.0:
            if self.ocr_model is None:
                self.ocr_model = load_ocr()  # Load on demand
            result = self.ocr_model(frame)
            self.last_ocr_time = time.time()
            return result
```

### Opción C: Distributed processing

```python
# Main device (Aria glasses)
- Capture: 30 FPS
- Lightweight detection: 18 FPS
- Stream to edge server: 10 FPS

# Edge server (laptop/desktop)
- Heavy processing: Segmentation, SLAM, LLM
- Return results: <200ms latency
```

---

## 📈 Roadmap de Escalabilidad

### Fase 1: Optimización actual (DONE ✅)
- [x] TensorRT FP16
- [x] Input resize
- [x] Non-blocking queues
- [x] SLAM opcional
- **Result:** 18 FPS, 1.3GB VRAM

### Fase 2: Añadir tracking básico (1 semana)
- [ ] Object tracking (SORT)
- [ ] Trajectory prediction
- [ ] Persistent IDs
- **Expected:** 16 FPS, 1.5GB VRAM

### Fase 3: Pose & OCR (2 semanas)
- [ ] MoveNet Lightning integration
- [ ] EasyOCR on-demand
- [ ] Gesture recognition
- **Expected:** 15 FPS, 1.8GB VRAM

### Fase 4: Hardware upgrade (si necesario)
- [ ] RTX 4060 12GB ($300)
- [ ] Coral TPU USB ($60)
- **Expected:** 25-30 FPS con todas las features

---

## ⚡ Optimizaciones para Nivel 2+

Si quieres añadir más sin degradar:

1. **Temporal sampling** (no procesar todo cada frame)
   ```python
   if frame_id % 2 == 0:
       run_pose_estimation()  # 15 FPS instead of 30
   if frame_id % 5 == 0:
       run_ocr()  # 6 FPS
   ```

2. **ROI processing** (solo procesar área de interés)
   ```python
   # Solo analizar centro de imagen para OCR
   roi = frame[h//4:3*h//4, w//4:3*w//4]
   text = ocr(roi)  # 4x faster
   ```

3. **Model distillation** (modelos más pequeños)
   - YOLOv8n → YOLOv8n-pruned (50% faster)
   - EasyOCR → Tesseract (CPU only)

4. **Async processing** (no bloquear main loop)
   ```python
   async def process_heavy(frame):
       await asyncio.to_thread(ocr.detect, frame)
   ```

---

## 🎓 Lecciones Aprendidas

1. **IPC overhead (20-25ms) es el real bottleneck**, no GPU
2. **Input resolution** tiene mayor impacto que modelo complexity
3. **VRAM es abundante** (79% libre), pero GPU cycles no
4. **Multiprocessing funciona mejor** que single-process para este caso
5. **Feature flags > always-on** para funcionalidades opcionales

---

## 🤔 Decisiones de Diseño

### ¿Cuándo añadir una feature?

**Checklist:**
- [ ] ¿Añade valor real al usuario final?
- [ ] ¿Cabe en budget de VRAM? (+X MB < 4.8GB disponible)
- [ ] ¿Mantiene >15 FPS? (Latencia <70ms)
- [ ] ¿Puede ser on-demand en lugar de always-on?
- [ ] ¿Hay alternativa más ligera? (MobileNet vs ResNet)

**Si 3+ respuestas son NO: NO añadir o buscar alternativa.**

---

## 📞 Contacto & Updates

Actualizado: 2025-11-24  
Sistema: aria-nav v1.0 (fase4-tensorrt)  
GPU: RTX 2060 6GB
