# 🏗️ Arquitectura del Sistema (High-Level)

Este diagrama representa el flujo de datos y la separación de procesos del **Aria Navigation System**.

```mermaid
graph TD
    %% Estilos
    classDef hardware fill:#f9f,stroke:#333,stroke-width:2px,color:#000;
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000;
    classDef gpu fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;
    classDef audio fill:#fff3e0,stroke:#ff6f00,stroke-width:2px,color:#000;

    subgraph Hardware [Hardware Layer]
        Aria["👓 Meta Aria Glasses"]:::hardware
        IMU["🧭 IMU Sensor"]:::hardware
    end

    subgraph MainProcess ["Main Process (Python)"]
        Observer["Observer (SDK Wrapper)"]
        Coord[Coordinator]:::process
        UI["Dashboard / UI"]
    end

    subgraph Workers [Multiprocessing Workers]
        direction TB
        
        subgraph CentralWorker ["Central Worker (GPU)"]
            Q_In_Central[Queue In]
            YOLO_Central["🚀 YOLOv8 (RGB)"]:::gpu
            Depth["📏 Depth Anything V2"]:::gpu
            Fusion[Data Fusion]
            Q_Out_Central[Queue Out]
        end

        subgraph SLAMWorker ["SLAM Worker (GPU)"]
            Q_In_Slam[Queue In]
            YOLO_Slam["🚀 YOLOv8 (Peripheral)"]:::gpu
            Q_Out_Slam[Queue Out]
        end
    end

    subgraph AudioLayer [Audio System]
        Decider["🧠 Decision Engine"]
        Router["🔀 Audio Router"]:::audio
        TTS["🗣️ TTS Engine"]:::audio
        Beeps["🔊 Spatial Beeps"]:::audio
    end

    %% Conexiones
    Aria -->|RGB Frames| Observer
    Aria -->|SLAM Frames| Observer
    IMU -->|Motion Data| Observer

    Observer -->|Raw Frames| Coord
    Coord -->|Put Frame| Q_In_Central
    Coord -->|Put Frame| Q_In_Slam

    %% Flujo Central
    Q_In_Central --> YOLO_Central
    Q_In_Central --> Depth
    YOLO_Central --> Fusion
    Depth --> Fusion
    Fusion -->|Detections + Depth| Q_Out_Central

    %% Flujo SLAM
    Q_In_Slam --> YOLO_Slam
    YOLO_Slam -->|Peripheral Detections| Q_Out_Slam

    %% Retorno a Main
    Q_Out_Central -->|Get Result| Coord
    Q_Out_Slam -->|Get Result| Coord

    %% Decisión y Audio
    Coord -->|Full State| Decider
    Decider -->|Candidate| Router
    Router -->|Critical| TTS
    Router -->|Spatial| Beeps

    %% UI
    Coord -->|Overlay Frame| UI
```

## 📝 Explicación del Flujo

1.  **Captura (Main Process):** El `Observer` extrae los frames de las gafas Aria usando el SDK oficial.
2.  **Distribución (Coordinator):** El `Coordinator` envía los frames a las colas de procesamiento (`Queue In`), separando la cámara central (RGB) de las periféricas (SLAM).
3.  **Procesamiento Paralelo (Workers):**
    *   **Central Worker:** Ejecuta YOLO (detección) y Depth-Anything (profundidad) en paralelo usando **CUDA Streams** en la GPU.
    *   **SLAM Worker:** Ejecuta un modelo YOLO más ligero para detectar amenazas laterales.
4.  **Fusión y Decisión:** Los resultados vuelven al `Coordinator`, que se los pasa al `Decision Engine`. Este aplica lógica de prioridades (¿Es una persona? ¿Está cerca?).
5.  **Feedback (Audio):** El `Audio Router` gestiona la salida para no saturar al usuario, mezclando pitidos espaciales (`Beeps`) con instrucciones verbales (`TTS`).
