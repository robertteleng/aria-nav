# Sistema de Navegación para Personas Ciegas con Gafas Aria

TFM desarrollando aplicación para navegación asistida usando gafas Meta Aria con detección de objetos en tiempo real y comandos de audio direccionales.

## 🎯 Estado Actual
✅ **Día 1 Completado:** RGB streaming + YOLO detección funcionando

### Funcionalidades Implementadas:
- [x] Stream RGB estable desde gafas Aria (60fps)
- [x] Detección de objetos en tiempo real (YOLOv11n)
- [x] Visualización con bounding boxes
- [x] Error handling robusto y cleanup limpio
- [x] Observer pattern para callbacks asíncronos
- [x] Rotación automática de imagen para orientación correcta

### En Desarrollo:
- [ ] Comandos de audio direccionales (izquierda/centro/derecha)
- [ ] Filtrado de objetos relevantes para navegación
- [ ] Cálculo de distancias con stereo depth
- [ ] Integración IMU para orientación
- [ ] Text-to-speech para feedback auditivo

## 🚀 Quick Start

### Prerequisitos
- macOS con Apple Silicon (recomendado)
- Conda/Miniconda instalado
- Gafas Meta Aria configuradas

### Instalación
```bash
# Clonar repositorio
git clone [tu-repo-url]
cd aria-navigation-tfm

# Crear environment
conda env create -f environment.yml
conda activate aria-navigation-tfm

# Verificar instalación
python --version  # Should be 3.10
```

### Uso Básico
```bash
# Ejecutar streaming básico con detección
cd src/
python aria_rgb_basic.py

# Controles:
# - 'q' o Ctrl+C para salir
# - Ventana redimensionable para mejor visualización
```

## 🏗️ Arquitectura

### Pipeline Actual (Día 1)
```
┌──────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Gafas    │───▶│ AriaRGB     │───▶│ YOLO        │───▶│ OpenCV      │
│ Aria     │    │ Observer    │    │ Detection   │    │ Display     │
│ (USB)    │    │             │    │ (YOLOv11n)  │    │             │
└──────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Arquitectura Objetivo Final
```
┌──────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Gafas    │───▶│ Vision  │───▶│ Spatial │───▶│ Audio   │───▶│ Usuario │
│ Aria     │    │ Process │    │ Analysis│    │ Commands│    │ Ciego   │
│          │    │ + YOLO  │    │ + IMU   │    │ + TTS   │    │         │
└──────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
```

## 📊 Tecnologías

### Core Dependencies
- **Python 3.10** - Lenguaje principal
- **Meta Aria SDK** - Interface con gafas
- **YOLOv11n** - Detección de objetos (Ultralytics)
- **OpenCV** - Procesamiento de imagen
- **NumPy** - Operaciones numéricas

### Development Tools
- **Git** - Control de versiones con ramas por funcionalidad
- **Conda** - Gestión de entornos
- **Notion** - Project management y documentación
- **RemNote** - Gestión de conocimiento técnico

## 🔧 Estructura del Proyecto

```
aria-navigation-tfm/
├── README.md                 # Este archivo
├── environment.yml          # Conda environment
├── .gitignore              # Git ignore rules
├── docs/
│   └── desarrollo_diario.md # Diario de desarrollo
├── src/
│   └── aria_rgb_basic.py   # Código principal (Día 1)
├── experiments/
│   └── meta_stream_all.py  # Código oficial Meta (referencia)
└── logs/                   # Outputs y debugging
```

## 🐛 Problemas Conocidos y Soluciones

### MPS Compatibility
- **Problema:** `torchvision::nms not implemented for MPS device`
- **Solución:** Usar CPU device para YOLO (performance aceptable)
- **Futuro:** Cambiar a MPS cuando Apple resuelva el bug

### Memory Layout
- **Problema:** `Image not contiguous` error en YOLO
- **Solución:** Aplicar `np.ascontiguousarray()` después de rotación

### Performance
- **Optimización:** YOLOv11n (nano) modelo para balance speed/accuracy
- **Configuración:** Profile28 para 60fps streaming

## 📈 Roadmap de Desarrollo

### ✅ Día 1 - Streaming Base
- [x] Setup proyecto y git workflow
- [x] RGB streaming desde Aria
- [x] Integración YOLO básica
- [x] Optimización performance

### 🔄 Día 2 - Audio Commands
- [ ] Text-to-speech integration
- [ ] Comandos direccionales básicos
- [ ] Filtrado de objetos por relevancia

### 🔄 Día 3 - Spatial Awareness
- [ ] Stereo depth calculation
- [ ] IMU data integration
- [ ] 3D position mapping

### 🔄 Semana 2 - Navigation Algorithm
- [ ] Path planning básico
- [ ] Obstacle avoidance
- [ ] User testing inicial

### 🔄 Evaluación Final
- [ ] Performance metrics
- [ ] User experience testing
- [ ] Documentación completa TFM

## 🧪 Testing y Desarrollo

### Configuración Aria
- **Interface:** USB (más estable que WiFi)
- **Profile:** profile28 (60fps)
- **Cámaras:** Solo RGB para Fase 1

### Performance Metrics
- **Latencia:** <200ms objetivo para comandos
- **FPS:** 60fps streaming, detección en tiempo real
- **CPU Usage:** ~30-40% con YOLOv11n en MacBook

## 🤝 Contribución y Desarrollo

### Git Workflow
```bash
# Crear nueva funcionalidad
git checkout dev
git checkout -b feature-name

# Desarrollar y commitear
git add .
git commit -m "feature-name: description"

# Merge cuando esté completo
git checkout dev
git merge feature-name
```

### Coding Standards
- Comentarios exhaustivos en español
- Docstrings para todas las funciones
- Error handling robusto
- Cleanup ordenado de recursos

## 📞 Contacto y Soporte

**Proyecto TFM 2025**
- **Autor:** [Tu nombre]
- **Universidad:** [Tu universidad]
- **Supervisor:** [Supervisor TFM]

---

**Última actualización:** Día 1 - 30/08/2025  
**Próxima milestone:** Audio commands integration  
**Estado:** ✅ Base funcional establecida, listo para Fase 2