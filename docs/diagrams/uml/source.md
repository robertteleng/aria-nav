## 1. Package Diagram - Organización Modular

**Propósito:** Muestra cómo organizaste tu código en módulos cohesivos.

**Estructura:**
- **Core:** Gestión básica (conexión Aria, manejo Ctrl+C)
- **Vision:** Todo lo relacionado con computer vision (YOLO, tracking)
- **Audio:** Sistema TTS y comandos direccionales  
- **Utils:** Herramientas auxiliares (visualización, configuración)
- **Observers:** Coordinador principal que integra todo

**Relaciones clave:**
- Observer usa (-->) los otros módulos
- DeviceManager registra Observer
- Todos leen Config (líneas punteadas = dependencia)

**Valor para TFM:** Demuestra arquitectura modular y separación de responsabilidades.

## 2. Class Diagram - Estructura Interna

**Propósito:** Detalle técnico de clases principales con métodos y atributos.

**Elementos importantes:**
- **Observer:** Coordinador central con threading asíncrono
- **YoloProcessor:** Lógica de detección y filtrado
- **AudioSystem:** Cooldown inteligente y TTS
- **DeviceManager:** Abstracción del hardware Aria

**Notaciones:**
- `-` = privado, `+` = público
- `*--` = composición (Observer contiene YoloProcessor)
- `-->` = asociación (DeviceManager usa Observer)
- `..>` = dependencia (crea objetos temporalmente)

**Valor para TFM:** Muestra diseño orientado a objetos y encapsulación.

## 3.1. Sequence - Main Flow

**Propósito:** Flujo temporal completo desde startup hasta output.

**Fases importantes:**
1. **Initialization:** DeviceManager conecta y configura
2. **Frame Loop:** Cada frame RGB procesado asíncronamente
3. **Processing:** YOLO → Audio → Render en paralelo
4. **Output:** Usuario recibe audio + pantalla actualizada

**Threading key:** 
- Callback `on_image_received` nunca se bloquea
- Processing real en hilo separado (`_processing_loop`)

**Valor para TFM:** Demuestra arquitectura tiempo real y no-blocking.

## 3.2. Sequence - Audio Processing

**Propósito:** Detalle del sistema anti-spam de audio.

**Lógica crítica:**
- **Phrase comparison:** Nueva frase = habla inmediato
- **Cooldown check:** Misma frase = esperar repeat_cooldown
- **Async TTS:** Subprocess no bloquea detección

**Innovación:** Tu cooldown inteligente (300ms→80ms latencia)

**Valor para TFM:** Muestra optimización UX específica para ciegos.

## 3.3. Sequence - YOLO Detection

**Propósito:** Pipeline de computer vision detallado.

**Pasos técnicos:**
1. **Inference:** YOLO model procesa frame
2. **Filtering:** Solo objetos navegacionales + confidence >0.6
3. **Spatial analysis:** Clasificación en zonas + distancia
4. **Relevance scoring:** Prioridad × confidence × tamaño
5. **Top selection:** Solo mejor detección (anti-spam)

**Valor para TFM:** Demuestra filtrado inteligente y scoring.