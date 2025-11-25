# 🤝 Contributing & Development Guide (Innovación)

> Flujo compacto para trabajar en Aria Navigation (build de investigación, no producción). Incluye ramas, commits, sync y pruebas mínimas.

## 🏁 En 30s
- Ramas: `main` estable; usa `feature/*`, `bugfix/*` o `docs/*` para cambios.
- Commits: Conventional Commits (`feat:`, `fix:`, `docs:`, `chore:`, `perf:`); pequeños y frecuentes.
- Workflow diario: desarrolla local, corre pruebas rápidas, sube rama y PR; si usas Jetson/Aria, sincroniza y smoke test allí.
- Pruebas mínimas: unit/integration relevantes + chequeos de performance si tocas pipeline.
- Referencias: performance en `guides/PERFORMANCE_OPTIMIZATION.md`, troubleshooting en `TROUBLESHOOTING.md`, archivo histórico en `archive/development/`.

## 🧭 Ramas y flujo Git
- `main`: rama estable.
- Feature/bug/docs: `feature/<nombre>`, `bugfix/<nombre>`, `docs/<nombre>`.
- Rebase vs merge: preferir rebase interactivo antes de abrir PR para mantener historia limpia.

### Mensajes de commit (Conventional Commits)
Usa prefijos: `feat`, `fix`, `docs`, `chore`, `perf`, `refactor`, `test`.
Ejemplos:
- `feat: add cooldown to audio router`
- `perf: enable yolo trt fp16 export`
- `docs: consolidate architecture and data flow`

## 🔄 Workflow diario
1) **Desarrollo local**
   - Edita en tu máquina (Mac/Linux). Si dependes de Aria/Jetson, usa mock para iterar.
2) **Pruebas rápidas**
   - Unit/integration relevantes (`pytest tests/...`); revisar `testing/README.md` para matrices y comandos.
   - Performance quick check si tocas el pipeline (ver guía de performance).
3) **Sync con hardware (si aplica)**
   - Sincroniza al Jetson/host de inferencia.
   - Smoke test del pipeline con modelos TensorRT/ONNX activos.
4) **Commits y PR**
   - Commits pequeños con prefijos; push a rama; abre PR describiendo impacto (funcional, perf, riesgos).

## 🧪 Testing mínimo recomendado
- **Visión/audio/spatial**: ejecuta las pruebas que cubran el módulo tocado.
- **Perf sensible**: medir FPS/latencias tras cambios en pipeline (usa telemetría y `PERFORMANCE_OPTIMIZATION.md`).
- **Audio**: si cambias router/tts, corre `testing/navigation_audio_testing.md` (guía) o smoke test con mock.
- **Mock vs hardware**: validar con mock primero; hardware para confirmación de rendimiento.

## 🛠️ Notas para entorno Jetson/Aria
- Mantén los modelos TensorRT/ONNX en caché; evita rebuild si no cambian.
- Verifica CUDA y drivers antes de probar; usa `nvidia-smi`/`tegrastats` según hardware.
- Si hay desfase, revisa frame skip y tamaño de entrada antes de tocar código.

## 🐛 Problemas comunes (ruta rápida)
- FPS bajo: revisa skips/config; consulta `guides/PERFORMANCE_OPTIMIZATION.md`.
- Audio lag/spam: revisa cooldown/colas; consulta `TROUBLESHOOTING.md` sección audio.
- Errores de sync hardware: re-sincroniza y valida dependencias en destino.

## 📚 Referencias
- Performance: `docs/guides/PERFORMANCE_OPTIMIZATION.md`
- Arquitectura y flujo: `docs/architecture/architecture_document.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`
- Archivo metodologías y frameworks: `docs/archive/development/`
