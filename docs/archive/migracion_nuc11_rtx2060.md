# Plan detallado de migración a Intel NUC 11 Enthusiast (RTX 2060)

Este documento describe los pasos recomendados para portar y optimizar **Aria Navigation System** en un Intel NUC 11 Enthusiast equipado con una **RTX 2060** dedicada. El objetivo es ganar rendimiento manteniendo reproducibilidad entre entornos.

> **Supuestos**  
> - Sistema operativo objetivo: Ubuntu 22.04 LTS (recomendado por soporte extendido y compatibilidad CUDA 12).  
> - Se dispone de acceso físico al NUC, permisos de administrador y capacidad para reinstalar firmware/OS si es necesario.

---

## 0. Preparación y línea base
1. **Captura de referencia actual**
   - Exporta `pip freeze > reports/requirements_$(date +%Y%m%d).txt` en la máquina origen.
   - Documenta versión de Python, PyTorch (`torch.__version__`), backends activos (MPS, CPU).
   - Guarda métricas de throughput actuales (fps, latencias `PROFILE` del pipeline) ejecutando `python src/main.py --profile` y archiva los resultados en `logs/baseline/`.
2. **Inventario de artefactos**
   - Lista de pesos y modelos (`yolo12n.pt`, checkpoints MiDaS/Depth Anything, assets de SLAM).
   - Configuraciones personalizadas (`.env`, `config.yaml`, credenciales del Aria SDK).
3. **Plan de reversión**
   - Define cómo restaurar el entorno original si la migración se retrasa (p.ej. snapshots del repo, backups de `logs/`).

---

## 1. Auditoría de compatibilidad
1. **Hardware**
   - Verifica BIOS/UEFI del NUC y habilita modo discreto para la RTX 2060 (desactiva gráficos híbridos si el driver lo requiere).
   - Confirma disponibilidad de puertos USB-C/USB-A necesarios para Meta Aria y ancho de banda para cámaras.
2. **Software crítico**
   - Comprueba que `projectaria-tools` y `aria-sdk` soportan Ubuntu 22.04 x86_64.
   - Revisa versiones de CUDA soportadas por PyTorch + RTX 2060 (>= CUDA 11.8 o 12.1 recomendado).
3. **Dependencias adicionales**
   - Valida compatibilidad de `ultralytics`, `opencv-python`, `transformers`, audio backend (PulseAudio/PipeWire).
   - Identifica paquetes macOS-only (ej. `say`) y define alternativas (`espeak`, `piper-tts`, etc.).

**Resultado esperado:** checklist sin bloqueos críticos. Documentar cualquier incompatibilidad y definir mitigaciones.

---

## 2. Automatización del entorno
1. **Scripts de aprovisionamiento**
   - Crea `scripts/setup_nuc.sh` que instale paquetes del sistema, controladores NVIDIA y dependencias de Python.
   - Aislado por secciones: `apt`, `nvidia`, `python`, `aria-sdk`.
2. **Gestión de entornos Python**
   - Define uso de `pyenv` + `poetry` o `conda` según preferencia. Para simplicidad, `python3.10-venv` + `requirements.txt` es suficiente si se congela versión.
   - Genera `requirements-linux.txt` usando `pip-tools` o adaptando `requirements-macos.txt` si existe.
3. **Infraestructura como código ligera**
   - Opcional: prepara playbook Ansible para instalar paquetes base y distribuir claves SSH.

---

## 3. Provisioning en el NUC
1. **Sistema operativo**
   - Actualiza BIOS/firmware.
   - Instala Ubuntu 22.04 LTS (modo UEFI, particionado con espacio para datasets/logs).
   - Ejecuta `sudo apt update && sudo apt upgrade -y`.
2. **Drivers NVIDIA y CUDA**
   - Instala driver recomendado: `sudo ubuntu-drivers autoinstall` o fija `nvidia-driver-535` (RTX 2060).
   - Reboot y valida con `nvidia-smi`.
   - Instala CUDA Toolkit (preferiblemente usando el repo oficial) y `cudnn` si se requiere.
3. **Dependencias del sistema**
   - `sudo apt install -y build-essential python3.10 python3.10-venv python3-pip ffmpeg libgl1-mesa-dev libopenblas-dev libsndfile1`
   - Instala herramientas de TTS: `sudo apt install -y espeak-ng` o integra piper/Coqui según calidad deseada.
4. **SDKs externos**
   - Sigue guías oficiales para `aria-sdk` (generalmente paquete `.deb` privado).
   - Instala `projectaria-tools` (`pip install projectaria-tools` o paquete `.deb`).

---

## 4. Configuración del proyecto
1. **Clonado y estructura**
   - Clona repo en `/opt/aria-navigation` o `${HOME}/aria-navigation`.
   - Copia archivos necesarios (`yolo12n.pt`, configs) usando `rsync` o `scp`.
2. **Entorno virtual**
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install --upgrade pip wheel setuptools`
   - `pip install -r requirements-linux.txt`
3. **Configuraciones específicas**
   - Ajusta `src/utils/config.py` para usar `device="cuda:0"` donde aplique (`YOLO_DEVICE`, `DEPTH_DEVICE`, etc.).
   - Habilita flags para perfiles de alto rendimiento (batching, pipeline concurrency) si disponibles.
4. **Soporte de audio**
   - Sustituye comando `say` por wrapper Linux (ej. `subprocess` con `espeak-ng`). Documenta cambios o abstrae en `AudioSystem`.

---

## 5. Migración de datos y artefactos
1. **Modelos y checkpoints**
   - Transfiere archivos grandes (`.pt`, `.onnx`, datasets) usando `rsync -av --progress`.
   - Verifica integridad con `sha256sum` antes/después.
2. **Logs e históricos**
   - Decide si migrar `logs/` completos o solo muestras para comparación.
3. **Credenciales y secretos**
   - Usa `sops`, `age` o `pass` para mover claves del Aria SDK, asegurando que no queden en texto plano.

---

## 6. Validación funcional
1. **Smoke tests**
   - Ejecuta `pytest -m "not slow"` (si existe marcador) para validar componentes básicos.
   - Corre `python src/main.py debug --headless` para verificar pipeline sin hardware.
2. **Pruebas con hardware real**
   - Conecta Meta Aria y ejecuta modo principal (`python src/main.py`).
   - Observa telemetría en `logs/audio_telemetry.jsonl` y verifica que `NavigationAudioRouter` funciona con TTS Linux.
3. **Verificación de SLAM y dashboards**
   - Confirma que `SlamDetectionWorker` corre en paralelo y que la salida en `PresentationManager` respeta el backend seleccionado (OpenCV/Rerun/Web).

---

## 7. Validación de rendimiento y optimización
1. **Benchmark comparativo**
   - Repite suite de benchmarks capturada en la fase 0, ahora en el NUC.
   - Registra FPS promedio, varianza y tiempos por etapa (`enhance`, `depth`, `yolo`, `nav_audio`, `render`).
2. **Ajustes CUDA**
   - Activa `torch.backends.cudnn.benchmark = True` si no hay dependencia de input shapes variables.
   - Ajusta `YOLO_IMG_SIZE`, `FRAME_SKIP`, `DEPTH_ANYTHING_VARIANT` buscando el mejor equilibrio.
3. **Termal y consumo**
   - Monitoriza con `nvidia-smi dmon` y `powermetrics` equivalente (p.ej. `tegrastats` no aplica pero se puede usar `nvtop`).
   - Configura alertas si la GPU alcanza temperaturas >80 °C.

---

## 8. Hardening y observabilidad
1. **Servicio systemd**
   - Crea `aria-navigation.service` para ejecutar el pipeline al boot (si aplica).
   - Incluye variables de entorno (`ARSDK_PATH`, `DATA_DIR`) y reinicio automático.
2. **Logging centralizado**
   - Redirige logs a `/var/log/aria-navigation/` con rotación (`logrotate`).
3. **Monitoreo continuo**
   - Integra `promtail`/`loki` o `node-exporter` si se requiere observabilidad remota.
   - Programa pruebas sintéticas (cron semanal) para garantizar que las dependencias siguen vigentes tras actualizaciones del sistema.

---

## 9. Checklist final
- [ ] Revisión de compatibilidad completada, sin bloqueos abiertos.
- [ ] Scripts de setup probados en NUC desde cero.
- [ ] Drivers NVIDIA/CUDA instalados y `nvidia-smi` operativo.
- [ ] Proyecto clonado y dependencias Python instaladas sin conflictos.
- [ ] Pipeline funcionando en modo debug y con hardware real.
- [ ] Benchmarks post-migración documentados y comparados contra línea base.
- [ ] Servicio/systemd o documentación operacional entregada.
- [ ] Backups y plan de soporte definidos.

---

## Próximos pasos sugeridos
1. **Automatizar benchmark**: crear script en `benchmarks/` que ejecute pipeline con inputs pregrabados y genere reporte comparativo.
2. **Abstraer TTS multiplataforma**: proveer interfaz unificada (`say`, `espeak`, `piper`) configurable en `config.py`.
3. **Documentar troubleshooting**: añadir sección en `docs/troubleshooting.md` con problemas comunes (drivers, permisos USB, sincronización reloj) específicos del NUC.

