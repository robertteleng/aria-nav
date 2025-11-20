# 🚀 Plan de Migración: Intel NUC 11 Enthusiast + RTX 2060

## 📊 Análisis del Proyecto Actual

### Hardware Actual
- **Plataforma**: macOS (Apple Silicon M1/M2)
- **Aceleración**: Apple Metal Performance Shaders (MPS)
- **Conectividad**: USB/WiFi con Meta Aria glasses
- **Performance Actual**: ~25-30 FPS con frame skipping agresivo

### Dependencias Hardware-Específicas Críticas
```python
# src/utils/config.py
YOLO_DEVICE = "mps"              # ❌ Específico de Apple
MIDAS_DEVICE = "mps"             # ❌ Específico de Apple  
YOLO_FORCE_MPS = True            # ❌ Específico de Apple
DEPTH_ENABLED = True             # ✅ Compatible
YOLO_FRAME_SKIP = 3              # ⚠️  Necesario por limitación MPS
DEPTH_FRAME_SKIP = 12            # ⚠️  Necesario por limitación MPS
```

### Componentes del Pipeline Actual
1. **ImageEnhancer**: CLAHE + gamma correction (CPU-bound, compatible)
2. **DepthEstimator**: MiDaS Small / Depth-Anything V2 (GPU-bound, optimizable)
3. **YoloProcessor**: YOLOv12n con frame skip (GPU-bound, optimizable)
4. **SlamDetectionWorker**: Visión periférica asíncrona (CPU-bound, compatible)
5. **AudioSystem**: TTS macOS `say` command (❌ incompatible con Linux)

---

## 🎯 Ventajas Esperadas con RTX 2060

### Performance Teórico
| Componente | Actual (MPS) | Esperado (RTX 2060) | Mejora |
|------------|--------------|---------------------|---------|
| YOLO Inference | ~33ms (30 FPS) | ~10-15ms (60-100 FPS) | **3-4x** |
| Depth Estimation | ~80ms (12 FPS) | ~20-30ms (30-50 FPS) | **3-4x** |
| SLAM Processing | ~125ms (8 FPS) | ~40-60ms (16-25 FPS) | **2-3x** |
| **Total Pipeline** | ~150-200ms | ~50-80ms | **3x** |

### Capacidades CUDA RTX 2060
- **CUDA Cores**: 1920
- **Tensor Cores**: 240 (para FP16 inference)
- **VRAM**: 6GB GDDR6 (vs ~8GB shared MPS)
- **TDP**: 160W (vs ~15-20W M1/M2)
- **CUDA Compute**: 7.5 (compatible con todas las librerías modernas)

### Optimizaciones Desbloqueadas
1. **Eliminar Frame Skipping**: Procesar cada frame (30 FPS → 30 FPS real)
2. **Modelos Más Grandes**: YOLOv12m/l, MiDaS Large, Depth-Anything Base
3. **Resolución Mayor**: 256px → 416-640px para YOLO
4. **TensorRT**: Cuantización INT8/FP16 para 2-3x adicional
5. **Batch Processing**: Procesar RGB + SLAM1 + SLAM2 en paralelo

---

## 🔧 Plan de Migración por Fases

### Fase 1: Preparación del Entorno (2-3 días)

#### 1.1 Sistema Base NUC
```bash
# Ubuntu 22.04 LTS recomendado (mejor soporte CUDA que 24.04)
sudo apt update && sudo apt upgrade -y
sudo apt install build-essential git cmake python3.10 python3-pip
```

#### 1.2 Instalación NVIDIA Driver + CUDA
```bash
# Driver NVIDIA (545+ recomendado)
ubuntu-drivers devices
sudo ubuntu-drivers autoinstall

# CUDA Toolkit 12.1+ (compatible con PyTorch 2.x)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-1

# Verificar
nvidia-smi
nvcc --version
```

#### 1.3 PyTorch con CUDA
```bash
# Crear entorno limpio
python3.10 -m venv .venv
source .venv/bin/activate

# PyTorch 2.2+ con CUDA 12.1
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# Verificar CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

#### 1.4 Dependencias del Proyecto
```bash
pip install ultralytics opencv-python numpy projectaria-tools transformers pytest

# Para Aria SDK (requiere distribución de Meta)
pip install aria-sdk-<version>.whl
```

#### 1.5 TTS en Linux (Sustituto de macOS `say`)
```bash
# Opción 1: espeak (ligero, calidad básica)
sudo apt install espeak
echo "Hola mundo" | espeak -v es

# Opción 2: pyttsx3 con espeak backend (actual en el código)
pip install pyttsx3

# Opción 3: Festival (mejor calidad, más pesado)
sudo apt install festival festvox-ellpc11k

# Opción 4: gTTS + mpg123 (cloud-based, requiere internet)
pip install gtts
sudo apt install mpg123
```

**Recomendación**: Usar `pyttsx3` con backend `espeak` para mantener compatibilidad del código actual.

---

### Fase 2: Adaptación del Código (3-4 días)

#### 2.1 Crear `mps_utils.py` → `device_utils.py`

**Archivo actual**: `src/core/vision/mps_utils.py` (específico MPS)

**Nuevo archivo**: `src/core/vision/device_utils.py` (multi-platform)

```python
"""Multi-platform device utilities for PyTorch (MPS, CUDA, CPU)"""
import os
import torch
from contextlib import contextmanager
from typing import Generator

def configure_torch_environment(force_cpu_fallback: bool = False) -> None:
    """Configure PyTorch for optimal performance on available hardware."""
    if torch.cuda.is_available():
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    elif torch.backends.mps.is_available():
        # MPS optimizations (macOS)
        if force_cpu_fallback:
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        else:
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

def get_preferred_device(preferred: str = "auto") -> torch.device:
    """Get best available device: CUDA > MPS > CPU."""
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    # Explicit device request
    if preferred.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    elif preferred.lower() == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def empty_device_cache() -> None:
    """Clear GPU cache for the active backend."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

@contextmanager
def autocast_inference(device: torch.device) -> Generator:
    """Enable automatic mixed precision for inference."""
    if device.type == "cuda":
        with torch.cuda.amp.autocast():
            yield
    else:
        yield  # No autocast for MPS/CPU
```

#### 2.2 Actualizar `config.py` para Multi-Platform

```python
# src/utils/config.py
import torch

class Config:
    # Auto-detect best device
    _DEVICE_AUTO = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Vision processing
    YOLO_DEVICE = _DEVICE_AUTO
    YOLO_FORCE_MPS = False  # Deprecated, use YOLO_DEVICE
    YOLO_CONFIDENCE = 0.50
    YOLO_IMAGE_SIZE = 416 if _DEVICE_AUTO == "cuda" else 256  # Aumentar en CUDA
    YOLO_FRAME_SKIP = 1 if _DEVICE_AUTO == "cuda" else 3     # Sin skip en CUDA
    YOLO_MAX_DETECTIONS = 20 if _DEVICE_AUTO == "cuda" else 8
    
    # Depth estimation  
    DEPTH_ENABLED = True
    DEPTH_BACKEND = "depth_anything_v2" if _DEVICE_AUTO == "cuda" else "midas"
    DEPTH_ANYTHING_VARIANT = "Base" if _DEVICE_AUTO == "cuda" else "Small"
    MIDAS_MODEL = "DPT_Large" if _DEVICE_AUTO == "cuda" else "MiDaS_small"
    MIDAS_DEVICE = _DEVICE_AUTO
    DEPTH_FRAME_SKIP = 2 if _DEVICE_AUTO == "cuda" else 12  # Mucho menos skip
    DEPTH_INPUT_SIZE = 384 if _DEVICE_AUTO == "cuda" else 256
    
    # SLAM peripheral vision
    PERIPHERAL_VISION_ENABLED = True
    SLAM_TARGET_FPS = 15 if _DEVICE_AUTO == "cuda" else 8
    SLAM_FRAME_SKIP = 2 if _DEVICE_AUTO == "cuda" else 12
    
    # TensorRT optimization (solo CUDA)
    TENSORRT_ENABLED = _DEVICE_AUTO == "cuda"
    TENSORRT_FP16 = True
    TENSORRT_CACHE_DIR = "./tensorrt_cache"
```

#### 2.3 Refactor `audio_system.py` para Linux

**Archivo actual**: `src/core/audio/audio_system.py`

```python
# Detección del sistema operativo
import platform
import subprocess

class AudioSystem:
    def __init__(self):
        self.platform = platform.system()
        
        if self.platform == "Darwin":  # macOS
            self._init_macos_tts()
        elif self.platform == "Linux":
            self._init_linux_tts()
        else:
            raise RuntimeError(f"Unsupported platform: {self.platform}")
    
    def _init_macos_tts(self):
        """Initialize macOS 'say' command."""
        self.tts_command = ["say", "-v", "Monica", "-r", str(Config.TTS_RATE)]
        print("[AudioSystem] Initialized macOS TTS (say)")
    
    def _init_linux_tts(self):
        """Initialize Linux TTS with pyttsx3/espeak."""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', Config.TTS_RATE)
            self.engine.setProperty('voice', 'spanish')  # o 'es' según backend
            self.use_pyttsx3 = True
            print("[AudioSystem] Initialized Linux TTS (pyttsx3+espeak)")
        except Exception as e:
            print(f"[WARN] pyttsx3 failed: {e}, falling back to espeak CLI")
            self.tts_command = ["espeak", "-v", "es", "-s", str(Config.TTS_RATE)]
            self.use_pyttsx3 = False
    
    def speak(self, text: str):
        """Speak text using platform-appropriate TTS."""
        if self.platform == "Darwin":
            subprocess.run(self.tts_command + [text], check=False)
        elif self.use_pyttsx3:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            subprocess.run(self.tts_command + [text], check=False)
```

#### 2.4 Añadir TensorRT Support (Opcional, +2-3x performance)

```python
# src/core/vision/tensorrt_converter.py
"""TensorRT conversion utilities for YOLO and depth models."""
from pathlib import Path
from ultralytics import YOLO

def convert_yolo_to_tensorrt(model_path: str, fp16: bool = True) -> str:
    """Convert YOLO model to TensorRT engine."""
    model = YOLO(model_path)
    
    # Export to TensorRT
    engine_path = model_path.replace('.pt', '_fp16.engine' if fp16 else '_fp32.engine')
    
    if not Path(engine_path).exists():
        print(f"[TensorRT] Converting {model_path} to TensorRT...")
        model.export(
            format='engine',
            half=fp16,
            simplify=True,
            workspace=4,  # 4GB workspace
            device=0
        )
        print(f"[TensorRT] ✓ Saved to {engine_path}")
    
    return engine_path

# Uso en yolo_processor.py
if Config.TENSORRT_ENABLED and self.device.type == "cuda":
    engine_path = convert_yolo_to_tensorrt(Config.YOLO_MODEL, Config.TENSORRT_FP16)
    self.model = YOLO(engine_path, task='detect')
else:
    self.model = YOLO(Config.YOLO_MODEL)
```

---

### Fase 3: Testing y Validación (2-3 días)

#### 3.1 Benchmarks Comparativos

```bash
# Ejecutar benchmarks en ambas plataformas
python benchmarks/benchmark_1_performance.py  # FPS y latencia
python benchmarks/benchmark_2_precision.py    # Precisión de detecciones
python benchmarks/benchmark_3_distances.py    # Estimación de profundidad

# Guardar resultados
mkdir -p benchmarks/results
mv logs/session_* benchmarks/results/nuc_rtx2060_baseline
```

**Métricas Objetivo con RTX 2060**:
- ✅ FPS promedio: **≥60 FPS** (vs 25-30 actual)
- ✅ Latencia máxima: **<50ms** (vs 150-200ms actual)
- ✅ Frame skip: **0** (vs 3-12 actual)
- ✅ Resolución YOLO: **416px** (vs 256px actual)
- ✅ Depth every frame: **2 frames** (vs 12 actual)

#### 3.2 Test de Estrés GPU

```bash
# Monitorear uso de GPU durante operación
nvidia-smi dmon -s u -c 1000

# Esperado:
# - GPU Utilization: 70-90%
# - Memory Usage: 3-4GB / 6GB
# - Temperature: 60-75°C
# - Power: 100-140W
```

#### 3.3 Validación Audio

```bash
# Test TTS Linux
python -c "from src.core.audio.audio_system import AudioSystem; a = AudioSystem(); a.speak('Persona adelante centro')"

# Verificar latencia audio
# Objetivo: <500ms desde detección hasta audio
```

---

### Fase 4: Optimizaciones Avanzadas (Opcional, 3-5 días)

#### 4.1 Batch Inference para SLAM

```python
# src/core/vision/slam_detection_worker.py
def process_slam_frames_batch(self, frames: List[np.ndarray]) -> List[YoloResult]:
    """Process multiple SLAM frames in parallel batch."""
    if not frames or not Config.TENSORRT_ENABLED:
        return [self.process_single_frame(f) for f in frames]
    
    # Stack frames para batch inference
    batch = np.stack(frames)  # [N, H, W, 3]
    
    # YOLO batch inference (mucho más eficiente en CUDA)
    results = self.yolo.predict(
        batch,
        conf=self.config.confidence,
        device=self.device,
        verbose=False
    )
    
    return results
```

#### 4.2 Async Pipeline con CUDA Streams

```python
# src/core/navigation/coordinator.py
import torch.cuda as cuda

class Coordinator:
    def __init__(self):
        if torch.cuda.is_available():
            self.stream_rgb = cuda.Stream()
            self.stream_depth = cuda.Stream()
            self.stream_slam = cuda.Stream()
    
    def process_frame_async(self, rgb_frame):
        """Process RGB, depth y SLAM en paralelo con CUDA streams."""
        with cuda.stream(self.stream_rgb):
            yolo_result = self.pipeline.detect_objects(rgb_frame)
        
        with cuda.stream(self.stream_depth):
            depth_map = self.pipeline.estimate_depth(rgb_frame)
        
        with cuda.stream(self.stream_slam):
            slam_events = self.slam_worker.get_detections()
        
        # Sincronizar antes de merge
        cuda.synchronize()
        
        return self._merge_results(yolo_result, depth_map, slam_events)
```

#### 4.3 Modelo de Profundidad Más Robusto

```python
# config.py
# Cambiar de MiDaS Small → Depth-Anything V2 Base
DEPTH_BACKEND = "depth_anything_v2"
DEPTH_ANYTHING_VARIANT = "Base"  # Small → Base → Large

# Performance esperado:
# - Base: ~25-30ms @ RTX 2060 (vs ~80ms Small @ MPS)
# - Calidad: +30% precisión en distancias <3m
```

---

## 📋 Checklist de Migración

### Pre-Migración
- [ ] Backup completo del repositorio actual
- [ ] Documentar configuración macOS actual (benchmarks baseline)
- [ ] Crear branch `migration/nuc-rtx2060`
- [ ] Lista de dependencias Python exactas: `pip freeze > requirements_macos.txt`

### Hardware Setup
- [ ] Instalar Ubuntu 22.04 LTS en NUC
- [ ] Instalar drivers NVIDIA 545+
- [ ] Instalar CUDA Toolkit 12.1+
- [ ] Verificar `nvidia-smi` y temperatura idle (<40°C)
- [ ] Test conectividad Aria glasses (USB/WiFi)

### Software Setup
- [ ] Python 3.10 + venv
- [ ] PyTorch 2.2+ con CUDA 12.1
- [ ] Ultralytics, OpenCV, NumPy
- [ ] projectaria-tools + aria-sdk
- [ ] TTS Linux (pyttsx3 + espeak)

### Código
- [ ] Refactor `mps_utils.py` → `device_utils.py`
- [ ] Actualizar `config.py` con detección automática
- [ ] Adaptar `audio_system.py` para Linux
- [ ] Eliminar/reducir frame skipping
- [ ] Aumentar resoluciones de inferencia
- [ ] (Opcional) Implementar TensorRT

### Testing
- [ ] Benchmark 1: Performance (FPS ≥60, latencia <50ms)
- [ ] Benchmark 2: Precision (sin regresión vs baseline)
- [ ] Benchmark 3: Depth accuracy (mejora esperada)
- [ ] Test audio Linux (latencia <500ms)
- [ ] Test SLAM peripheral vision (15+ FPS)
- [ ] Test largo (1h+ sin memory leaks)

### Validación Final
- [ ] Prueba en entorno real (indoor + outdoor)
- [ ] Validar cooldowns de audio (sin solapamiento)
- [ ] Verificar telemetría (logs completos)
- [ ] Dashboard OpenCV/Rerun funcional
- [ ] Temperatura GPU estable (<80°C en carga)

---

## 🚨 Riesgos y Mitigaciones

### Riesgo 1: Driver NVIDIA Incompatible
**Probabilidad**: Media  
**Impacto**: Alto  
**Mitigación**: 
- Usar Ubuntu 22.04 LTS (mejor compatibilidad que 24.04)
- Driver 545+ (no bleeding edge)
- Rollback a driver anterior si falla

### Riesgo 2: TTS Linux Peor Calidad
**Probabilidad**: Alta  
**Impacto**: Medio  
**Mitigación**:
- Test múltiples backends (espeak, festival, piper)
- Considerar TTS cloud (gTTS, Amazon Polly) si calidad crítica
- Ajustar rate/pitch para inteligibilidad

### Riesgo 3: Aria SDK Incompatible con Linux
**Probabilidad**: Baja  
**Impacto**: Crítico  
**Mitigación**:
- Verificar documentación oficial Meta Aria (soporte Linux existe)
- Test streaming USB/WiFi en NUC antes de migración completa
- Fallback: usar Mac como Aria gateway + NUC para inferencia (modo híbrido)

### Riesgo 4: Memory Leak en CUDA
**Probabilidad**: Media  
**Impacto**: Medio  
**Mitigación**:
- Llamar `torch.cuda.empty_cache()` periódicamente
- Monitorear VRAM con `nvidia-smi`
- Implementar reinicio automático si VRAM >90%

### Riesgo 5: Performance No Mejora lo Esperado
**Probabilidad**: Baja  
**Impacto**: Alto  
**Mitigación**:
- Verificar que PyTorch usa CUDA (no CPU fallback)
- Habilitar cuDNN benchmark
- Considerar TensorRT si no alcanza 60 FPS
- Profiling con `torch.profiler` para identificar cuellos de botella

---

## 💰 Costo-Beneficio

### Inversión Estimada
- **Hardware**: Intel NUC 11 + RTX 2060 (€800-1200 usado/reacondicionado)
- **Tiempo desarrollo**: 7-12 días persona
- **Riesgo**: Bajo-Medio (tecnología probada)

### Retorno Esperado
- **Performance**: 3-4x mejora en FPS (25 → 60-100 FPS)
- **Calidad**: Modelos más grandes/precisos sin penalización
- **Latencia**: 3x reducción (150ms → 50ms)
- **User Experience**: Audio más responsive, detecciones más suaves
- **Escalabilidad**: Base para features futuros (tracking multi-persona, reconocimiento facial, SLAM dense)

### Alternativas Consideradas
1. **Mantener macOS + eGPU**: Costoso (€500+), soporte limitado Apple Silicon
2. **Jetson AGX Orin**: Más portátil pero ~2x más caro, menos potencia que RTX 2060
3. **Cloud Inference**: Latencia inaceptable (>500ms), requiere internet

**Recomendación**: ✅ **Proceder con NUC + RTX 2060**

---

## 📚 Recursos Adicionales

### Documentación Técnica
- [PyTorch CUDA Best Practices](https://pytorch.org/docs/stable/notes/cuda.html)
- [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Ultralytics Export Formats](https://docs.ultralytics.com/modes/export/)
- [Meta Aria SDK Linux Setup](https://developers.meta.com/aria)

### Benchmarks de Referencia
- YOLOv11n @ RTX 2060: ~8-12ms (80-120 FPS) @ 640px
- Depth-Anything-V2-Base @ RTX 2060: ~20-30ms @ 384px
- Total pipeline esperado: ~40-60ms (20-25 FPS end-to-end real)

### Comunidad y Soporte
- [Ultralytics Discord](https://discord.gg/ultralytics) - YOLO optimization
- [PyTorch Forums](https://discuss.pytorch.org/) - CUDA issues
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/) - TensorRT

---

## 🎯 Próximos Pasos Inmediatos

1. **Validar Hardware** (1 día):
   - Confirmar NUC 11 soporta RTX 2060 (PCIe 4.0 x16, PSU suficiente)
   - Verificar disipación térmica (RTX 2060 = 160W TDP)

2. **Setup Base** (2 días):
   - Instalar Ubuntu 22.04 + drivers
   - Instalar CUDA + PyTorch
   - Test `nvidia-smi` y `torch.cuda.is_available()`

3. **Migración Código** (3-4 días):
   - Implementar `device_utils.py`
   - Adaptar `config.py` y `audio_system.py`
   - Test unitarios básicos

4. **Benchmarking** (2 días):
   - Ejecutar suite completa
   - Comparar con baseline macOS
   - Documentar resultados

5. **Optimización** (Opcional, 3-5 días):
   - TensorRT conversion
   - Batch inference SLAM
   - CUDA streams async

**Tiempo Total Estimado**: 7-15 días (según profundidad de optimizaciones)

---

## ✅ Conclusión

La migración al Intel NUC 11 + RTX 2060 es **altamente recomendable** para este proyecto:

- ✅ **Performance**: Mejora esperada de 3-4x en throughput
- ✅ **Viabilidad**: Riesgo técnico bajo, stack probado
- ✅ **Costo**: Razonable para el beneficio (~€1000 total)
- ✅ **Escalabilidad**: Base sólida para features avanzados
- ✅ **Compatibilidad**: PyTorch/CUDA/Linux stack maduro

**Recomendación final**: Proceder con migración por fases según plan descrito.
