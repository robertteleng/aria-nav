# 🏗️ Arquitectura Detallada - TFM Sistema Navegación para Ciegos con Gafas Aria

## 📋 **Información del Proyecto**
- **Nombre**: Sistema de Navegación Asistida para Personas con Discapacidad Visual
- **Hardware**: Meta Aria Glasses
- **Estado Actual**: Día 2 completado
- **Objetivo**: Sistema de navegación en tiempo real con audio direccional

---

## 🎯 **Visión General del Sistema**

### **Objetivo Principal**
Desarrollar un sistema de navegación en tiempo real que utilice computer vision y audio direccional para asistir a personas con discapacidad visual en entornos urbanos y domésticos.

### **Casos de Uso Principales**
1. **Navegación urbana**: Detectar peatones, vehículos, señales de tráfico
2. **Navegación indoor**: Identificar obstáculos, puertas, escaleras
3. **Interacción social**: Reconocer personas y gestos básicos
4. **Seguridad**: Alertas de peligros inmediatos

---

## 🏛️ **Arquitectura del Sistema**

```
┌─────────────────────────────────────────────────────────────┐
│                    ARIA GLASSES                             │
├─────────────────┬─────────────────┬─────────────────────────┤
│   CÁMARA RGB    │   MICRÓFONOS    │    ALTAVOCES            │
│   (640x480)     │   (Stereo)      │    (Audio Direccional)  │
└─────────────────┴─────────────────┴─────────────────────────┘
         │                 │                     ▲
         ▼                 ▼                     │
┌─────────────────────────────────────────────────────────────┐
│                  PROCESSING UNIT                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────────────┐ ┌──────────────┐ │
│  │  VISION MODULE  │  │  AUDIO MODULE    │ │ NAVIGATION   │ │
│  │                 │  │                  │ │ MODULE       │ │
│  │ • YOLOv11n      │  │ • TTS Engine     │ │ • Path Plan  │ │
│  │ • Object Track  │  │ • Audio Queue    │ │ • Obstacles  │ │
│  │ • Zone Detect   │  │ • Spatial Audio  │ │ • Priorities │ │
│  └─────────────────┘  └──────────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧩 **Componentes Principales**

### **1. Vision Processing Module**
```python
class VisionModule:
    ├── YOLOv11n Detection Engine
    ├── Object Tracking System  
    ├── Zone Classification (Left/Center/Right)
    ├── Distance Estimation
    └── Detection Buffer & Smoothing
```

**Responsabilidades:**
- Captura de video en tiempo real (30-60 FPS)
- Detección de objetos con YOLOv11n
- Seguimiento temporal de objetos
- Clasificación espacial en zonas
- Estimación de distancia relativa

**Input**: RGB Video Stream (640x480)
**Output**: Detection Objects con posición y metadata

### **2. Audio Processing Module**
```python
class AudioModule:
    ├── Text-to-Speech Engine (pyttsx3)
    ├── Audio Message Queue
    ├── Spatial Audio Processing
    ├── Priority Management
    └── Threading System
```

**Responsabilidades:**
- Conversión texto a voz en tiempo real
- Cola de mensajes sin solapamiento
- Procesamiento de audio espacial
- Control de prioridades y cooldowns
- Ejecución asíncrona no bloqueante

**Input**: Detection Objects + Spatial Data
**Output**: Audio Commands direccionales

### **3. Navigation Intelligence Module**
```python
class NavigationModule:
    ├── Object Priority System
    ├── Spatial Reasoning
    ├── Safety Alert System
    ├── Context Awareness
    └── User Preference Engine
```

**Responsabilidades:**
- Priorización inteligente de objetos
- Análisis de contexto espacial
- Generación de alertas de seguridad
- Adaptación a preferencias de usuario
- Toma de decisiones de navegación

---

## 📊 **Flujo de Datos Detallado**

### **Pipeline Principal**
```
1. [CAMERA] → RGB Frame (640x480, 30fps)
2. [YOLO] → Object Detections (bbox, class, confidence)  
3. [PROCESSING] → Spatial Analysis (zone, distance, priority)
4. [FILTERING] → Relevant Objects Only
5. [INTELLIGENCE] → Priority Ranking & Context
6. [AUDIO] → TTS Message Generation
7. [OUTPUT] → Spatial Audio Commands
```

### **Flujo Temporal**
```
Frame N:   Capture → Detect → Process → Queue Audio
Frame N+1: Capture → Detect → Process → Queue Audio
...
Audio Thread: [Continuous] → Play Queued Messages
```

---

## 🎯 **Objetos de Navegación y Prioridades**

### **Clasificación por Importancia**

#### **🔴 Críticos (Prioridad 8-10)**
```python
critical_objects = {
    'person': {'priority': 10, 'spanish': 'persona'},
    'stop sign': {'priority': 9, 'spanish': 'señal de stop'},
    'car': {'priority': 8, 'spanish': 'coche'},
    'truck': {'priority': 8, 'spanish': 'camión'},
    'bus': {'priority': 8, 'spanish': 'autobús'}
}
```

#### **🟡 Importantes (Prioridad 5-7)**
```python
important_objects = {
    'bicycle': {'priority': 7, 'spanish': 'bicicleta'},
    'motorcycle': {'priority': 7, 'spanish': 'motocicleta'},
    'traffic light': {'priority': 6, 'spanish': 'semáforo'},
    'stairs': {'priority': 5, 'spanish': 'escaleras'}
}
```

#### **🟢 Contextuales (Prioridad 1-4)**
```python
contextual_objects = {
    'door': {'priority': 4, 'spanish': 'puerta'},
    'chair': {'priority': 3, 'spanish': 'silla'},
    'bench': {'priority': 2, 'spanish': 'banco'}
}
```

### **Sistema de Modificadores de Prioridad**
```python
priority_modifiers = {
    'distance': {
        'muy_cerca': 2.0,    # Multiplicador x2
        'cerca': 1.5,        # Multiplicador x1.5  
        'lejos': 1.0         # Sin modificador
    },
    'position': {
        'centro': 1.3,       # +30% prioridad
        'izquierda': 1.0,    # Sin modificador
        'derecha': 1.0       # Sin modificador
    }
}
```

---

## 🔧 **Configuraciones Técnicas**

### **Vision Processing**
```yaml
vision_config:
  model: "yolov11n.pt"
  input_resolution: [640, 480]
  target_fps: 30
  confidence_threshold: 0.5
  nms_threshold: 0.45
  detection_buffer_size: 5
```

### **Audio System**
```yaml
audio_config:
  tts_engine: "pyttsx3"
  speech_rate: 150  # WPM
  volume: 0.9
  voice_language: "es"
  queue_max_size: 3
  announcement_cooldown: 2.0  # seconds
```

### **Spatial Processing**
```yaml
spatial_config:
  zones:
    left: [0, 213]      # pixels
    center: [213, 426]  # pixels  
    right: [426, 640]   # pixels
  distance_estimation:
    person_close: 200   # bbox height pixels
    person_medium: 100  # bbox height pixels
    car_close: 150      # bbox height pixels
    car_medium: 75      # bbox height pixels
```

---

## ⚡ **Performance y Optimizaciones**

### **Métricas Objetivo**
```yaml
performance_targets:
  detection_fps: 30-60
  audio_latency: <1000ms
  memory_usage: <2GB
  cpu_usage: <70%
  battery_life: >4hours
```

### **Optimizaciones Implementadas**

#### **Vision Optimizations**
- **YOLOv11n**: Modelo nano para máximo performance
- **Resolution**: 640x480 balanceando calidad/velocidad
- **Buffer smoothing**: Reduce detecciones falsas
- **Confidence filtering**: Solo objetos >50% confianza

#### **Audio Optimizations**  
- **Threading**: Audio no bloquea detección
- **Queue system**: Evita solapamiento de mensajes
- **Cooldown**: Previene spam de anuncios
- **Priority queue**: Mensajes importantes primero

#### **Memory Optimizations**
- **Detection buffer**: Circular buffer tamaño fijo
- **Model caching**: YOLO se carga una vez
- **Garbage collection**: Limpieza automática de objetos

---

## 🛣️ **Plan de Desarrollo por Días**

### **✅ Día 1 - Base Foundation (COMPLETADO)**
- [x] Setup básico de captura de video
- [x] Integración YOLOv11n
- [x] Detección en tiempo real funcionando
- [x] Performance optimizado a 60fps

### **✅ Día 2 - Audio Direccional (COMPLETADO)**  
- [x] Integración Text-to-Speech
- [x] Sistema de zonas espaciales
- [x] Audio queue y threading
- [x] Priorización de objetos
- [x] Mensajes contextuales

### **🔄 Día 3 - Audio Espacial 3D (EN PROGRESO)**
- [ ] Implementar audio posicional
- [ ] Calibración de distancia mejorada  
- [ ] Navegación turn-by-turn básica
- [ ] Testing en entornos reales

### **📅 Días Futuros - Funcionalidades Avanzadas**
- **Día 4**: Reconocimiento de gestos y personas
- **Día 5**: Integración GPS y mapas
- **Día 6**: Machine Learning personalizado
- **Día 7**: Optimización para Aria hardware

---

## 🧪 **Testing y Validación**

### **Unit Tests**
```python
test_modules = [
    "test_vision_detection",
    "test_audio_generation", 
    "test_spatial_processing",
    "test_priority_system",
    "test_performance_metrics"
]
```

### **Integration Tests**
```python
integration_scenarios = [
    "indoor_navigation",
    "outdoor_crosswalk",
    "crowded_environment", 
    "low_light_conditions",
    "multiple_audio_sources"
]
```

### **User Testing Criteria**
- **Accuracy**: >90% detecciones correctas
- **Latency**: <1s respuesta audio
- **Usability**: Comprensión >95% mensajes
- **Safety**: 0 falsos negativos críticos

---

## 📱 **Deployment Architecture**

### **Hardware Requirements**
```yaml
aria_glasses:
  cpu: ARM64 processor
  ram: 4GB minimum
  storage: 32GB
  cameras: RGB + Depth sensors
  audio: Stereo speakers + microphones
  connectivity: WiFi + Bluetooth
  battery: 8+ hours
```

### **Software Stack**
```yaml
software_stack:
  os: Android/Linux embedded
  python: 3.8+
  frameworks:
    - OpenCV 4.5+
    - Ultralytics YOLO
    - PyTorch Mobile
    - pyttsx3
  dependencies:
    - numpy
    - threading
    - queue
    - time
```

---

## 🔒 **Consideraciones de Privacidad y Seguridad**

### **Privacy by Design**
- **Local Processing**: Todo el procesamiento en device
- **No Cloud**: Sin envío de video/audio a servidores
- **Encrypted Storage**: Configuraciones usuario encriptadas
- **Minimal Data**: Solo almacenar preferencias esenciales

### **Safety Measures**
- **Fail-Safe Audio**: Alertas críticas nunca se pierden
- **Battery Monitoring**: Avisos de batería baja
- **Performance Monitoring**: Degradación automática si necesario
- **Emergency Protocols**: Procedimientos de emergencia

---

## 📈 **Roadmap Futuro**

### **Versión 2.0 - Advanced Features**
- **Computer Vision**: Reconocimiento facial, lectura de texto
- **AI Assistant**: Interacción por voz natural
- **Social Integration**: Compartir rutas y puntos de interés
- **Health Monitoring**: Métricas de actividad y movilidad

### **Versión 3.0 - Ecosystem**
- **Smart City Integration**: Conexión con semáforos inteligentes
- **AR Overlay**: Realidad aumentada para usuarios con visión parcial
- **Multi-device**: Sincronización con smartphone y smartwatch
- **Community Features**: Red de usuarios para feedback colaborativo

---

## 📚 **Referencias Técnicas**

### **Papers y Research**
- YOLOv11 Architecture and Performance Analysis
- Real-time Object Detection for Assistive Technology
- Spatial Audio Processing for Navigation Systems
- Computer Vision Applications in Accessibility

### **Frameworks y Libraries**
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [pyttsx3](https://pypi.org/project/pyttsx3/)
- [Meta Aria SDK](https://www.meta.com/smart-glasses/)

---

**📊 Estado del Proyecto**: 2/7 días completados  
**🎯 Próximo Milestone**: Audio espacial 3D  
**📅 Última actualización**: Día 2 - Sistema audio direccional  
**✅ Performance actual**: 30-60fps, <1s latencia audio
