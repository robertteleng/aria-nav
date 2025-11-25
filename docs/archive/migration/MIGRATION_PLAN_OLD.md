# Plan de Migración: de macOS (MPS) a Intel NUC (NVIDIA CUDA)

Este documento detalla el plan para migrar el proyecto `aria-navigation` desde un entorno de desarrollo en macOS que utiliza la aceleración gráfica de Apple (MPS) a un sistema Intel NUC 11 Enthusiast equipado con una GPU NVIDIA RTX 2060, utilizando el ecosistema CUDA para obtener el máximo rendimiento.

## 1. Hardware y Software de Destino

- **Hardware**: Intel NUC 11 Enthusiast
- **GPU**: NVIDIA RTX 2060
- **Sistema Operativo**: Ubuntu 22.04 LTS
- **Tecnología de Aceleración**: NVIDIA CUDA Toolkit y cuDNN

---

## Fase 1: Análisis y Preparación del Entorno

El objetivo de esta fase es preparar el nuevo hardware e identificar todo el software y código que necesitará ser modificado.

### 1.1. Configuración del Sistema en el NUC

1.  **Instalar Sistema Operativo**:
    *   Instala **Ubuntu 22.04 LTS** en el Intel NUC. Es el estándar de facto para robótica y aplicaciones de IA.

2.  **Instalar Controladores NVIDIA**:
    *   Una vez instalado Ubuntu, abre una terminal y ejecuta el siguiente comando para instalar automáticamente los drivers recomendados:
        ```bash
        sudo ubuntu-drivers autoinstall
        ```
    *   Reinicia el sistema y verifica que los drivers están activos con el comando:
        ```bash
        nvidia-smi
        ```
    *   Este comando debería mostrar una tabla con información sobre la GPU RTX 2060.

3.  **Instalar CUDA Toolkit y cuDNN**:
    *   Visita la [web oficial de NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) y selecciona las opciones para Ubuntu 22.04.
    *   Sigue las instrucciones de instalación que proporciona la web. Esto es **crítico** para que PyTorch y otras librerías puedan usar la GPU.
    *   Instala la librería **cuDNN** siguiendo su [guía de instalación](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) para mejorar el rendimiento de las redes neuronales.

### 1.2. Gestión de Dependencias

1.  **Generar `requirements.txt`**:
    *   En tu Mac, activa el entorno virtual de tu proyecto y ejecuta:
        ```bash
        pip freeze > requirements.txt
        ```
    *   Añade este fichero al repositorio de Git.

2.  **Revisar Dependencias**:
    *   Inspecciona el `requirements.txt`. La dependencia más importante es `torch`. Su instalación en el NUC será diferente para asegurar que se instala con soporte para CUDA.

### 1.3. Identificación de Código Específico de Plataforma

1.  **Código de Aceleración (MPS)**:
    *   El fichero `src/core/vision/mps_utils.py` es específico de Apple.
    *   Es necesario buscar todas las importaciones y usos de este módulo en el proyecto para reemplazarlos. Un comando útil para esto es:
        ```bash
        grep -r "mps_utils" src/
        ```

2.  **Código de Acceso a Hardware**:
    *   Ficheros como `src/core/hardware/device_manager.py` pueden contener lógica para acceder a la cámara o sensores IMU que es específica de macOS. En Linux, los dispositivos se nombran de forma diferente (p. ej., `/dev/video0`). Este código podría necesitar ajustes.

---

## Fase 2: Migración del Código y Dependencias

Esta fase se centra en adaptar el software al nuevo entorno.

### 2.1. Replicación del Entorno Python en el NUC

1.  **Clonar el Repositorio**:
    *   Clona tu proyecto de Git en el NUC.

2.  **Crear Entorno Virtual**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instalar Dependencias para CUDA**:
    *   Primero, instala PyTorch con soporte para CUDA. La versión de `cuXXX` debe coincidir con la versión del CUDA Toolkit que instalaste. Por ejemplo, para CUDA 11.8:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    *   A continuación, instala el resto de dependencias. Es recomendable eliminar la línea de `torch` del `requirements.txt` para evitar conflictos.
        ```bash
        pip install -r requirements.txt
        ```

### 2.2. Adaptación del Código de MPS a CUDA

1.  **Reemplazar Dispositivo de Cómputo**:
    *   El cambio principal en los ficheros de procesamiento de visión (como `yolo_processor.py`) será cambiar la selección del dispositivo.
    *   **Código antiguo (aproximado)**:
        ```python
        # import mps_utils
        # device = mps_utils.get_device()
        device = 'mps'
        ```
    *   **Código nuevo**:
        ```python
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ```

2.  **Eliminar `mps_utils.py`**:
    *   Una vez que todas las referencias a `mps_utils.py` hayan sido eliminadas y reemplazadas por la lógica de CUDA, el fichero puede ser borrado.

### 2.3. Ajustes de Acceso a Hardware

1.  **Configuración de Cámara**:
    *   Si usas OpenCV, es posible que necesites cambiar el índice de la cámara.
    *   **Código antiguo (ejemplo)**: `cap = cv2.VideoCapture(0)`
    *   **Código nuevo**: Puede ser el mismo, pero podría cambiar a `1` o a una ruta de dispositivo si hay varias cámaras. Se debe probar empíricamente.

---

## Fase 3: Pruebas y Validación

El objetivo es asegurar que el proyecto funciona correctamente en el nuevo hardware.

### 3.1. Pruebas Unitarias

*   Ejecuta toda la suite de tests existente para verificar que la lógica central no se ha roto durante la migración.
    ```bash
    pytest tests/
    ```
*   Corrige cualquier test que falle antes de continuar.

### 3.2. Pruebas End-to-End

*   Ejecuta la aplicación principal:
    ```bash
    python src/main.py
    ```
*   **Lista de Verificación Manual**:
    - [ ] ¿Se muestra correctamente la imagen de la cámara?
    - [ ] ¿El modelo YOLO detecta objetos y dibuja los bounding boxes?
    - [ ] ¿El sistema de audio genera las señales correctamente en función de las detecciones?
    - [ ] ¿Se generan los logs de telemetría sin errores?
    - [ ] ¿La aplicación responde de forma fluida?

---

## Fase 4: Optimización y Benchmarking

Ahora que el proyecto funciona, mediremos y optimizaremos su rendimiento.

### 4.1. Benchmarking Comparativo

*   Ejecuta los scripts de `benchmarks/` tanto en el Mac como en el NUC.
    ```bash
    python benchmarks/benchmark_1_performance.py
    # ... y los demás benchmarks
    ```
*   Documenta los resultados (FPS, latencia, precisión) en una tabla para cuantificar la mejora.

### 4.2. Monitorización de la GPU

*   Mientras la aplicación se ejecuta, usa `nvidia-smi` para observar el comportamiento de la GPU en tiempo real.
    ```bash
    watch -n 1 nvidia-smi
    ```
*   **Puntos a observar**:
    *   **GPU-Util**: Debería ser alto durante el procesamiento de imágenes. Si es bajo, la GPU no se está aprovechando.
    *   **Memory-Usage**: Para asegurar que el modelo cabe en la VRAM de la RTX 2060.

### 4.3. Optimización Avanzada (Siguientes Pasos)

*   **TensorRT**: Para exprimir al máximo la GPU, investiga la posibilidad de convertir tu modelo `.pt` a un motor de TensorRT. Esto puede duplicar o triplicar la velocidad de inferencia, pero requiere un paso de conversión adicional.
*   **Precisión Mixta (AMP)**: Usa `torch.cuda.amp` para acelerar el entrenamiento y la inferencia utilizando tipos de datos de 16 bits, lo que reduce el uso de memoria y puede aumentar la velocidad.
