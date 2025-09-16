#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mac client: captura frames de Aria (vía DeviceManager) y los envía al Jetson
- Usa ImageZMQ (REQ/REP) con envío en JPG para reducir carga
- Autocura el socket si entra en estado inválido (Operation cannot be accomplished...)
- Logs de estado y métricas
"""

import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import imagezmq

# Hacemos visible el paquete del repo (core, utils, ...)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hardware.device_manager import DeviceManager  # tu clase existente
from utils.ctrl_handler import CtrlCHandler            # tu clase existente


class MacSender:
    def __init__(self, jetson_ip: str = "jetson-desktop.local", port: int = 5555, fps: int = 30):
        self.jetson_ip = jetson_ip
        self.port = port
        self.fps = fps

        self.endpoint = f"tcp://{self.jetson_ip}:{self.port}"
        print(f"[MAC] 🚀 Endpoint ZMQ: {self.endpoint}")

        self.sender = self._make_sender()  # ImageZMQ ImageSender (REQ)
        self.device_manager = DeviceManager()

        # Stats
        self.frames_recv_total = 0      # frames totales recibidos de Aria
        self.frames_rgb_total = 0       # frames RGB filtrados
        self.frames_sent_total = 0      # frames enviados al Jetson
        self.last_stat_time = time.time()

    def _make_sender(self):
        try:
            s = imagezmq.ImageSender(connect_to=self.endpoint)
            print("[MAC] ✅ ImageSender creado")
            return s
        except Exception as e:
            print(f"[MAC] ❌ No se pudo crear ImageSender: {e}")
            raise

    # ========================= Aria callbacks =========================
    def on_image_received(self, image, record):
        """Callback llamado por DeviceManager al recibir un frame."""
        self.frames_recv_total += 1

        # Identificar cámara RGB
        camera_type = None
        if hasattr(record.camera_id, "name"):
            name = record.camera_id.name.lower()
            if "rgb" in name:
                camera_type = "rgb"
        else:
            # Algunos SDK exponen int (0,1,214...) — mapea si lo necesitas
            if record.camera_id in (0, 1, 214, 3):
                camera_type = "rgb"

        if self.frames_recv_total <= 10:
            dbg = record.camera_id.name if hasattr(record.camera_id, "name") else record.camera_id
            print(f"[MAC] 📷 Recibido frame #{self.frames_recv_total} cam={dbg} → tipo={camera_type}")

        if camera_type != "rgb":
            return  # ignoramos no-RGB

        self.frames_rgb_total += 1

        # Rotar si tus gafas lo requieren (ajusta si hace falta)
        frame = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # Codificar a JPG (menos ancho de banda y CPU en Jetson)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            print("[MAC] ⚠️ imencode falló; frame descartado")
            return

        # Enviar y leer reply (REQ/REP)
        try:
            t0 = time.time()
            reply = self.sender.send_jpg("mac_aria", jpg)  # <- bloquea hasta reply
            dt = (time.time() - t0) * 1000.0

            self.frames_sent_total += 1
            if self.frames_sent_total <= 5:
                print(f"[MAC] 📤 Enviado #{self.frames_sent_total} ({dt:.1f} ms) reply={reply}")

            # métricas cada 100 envíos
            if self.frames_sent_total % 100 == 0:
                now = time.time()
                span = now - self.last_stat_time
                fps = 100.0 / span if span > 0 else 0.0
                print(f"[MAC] 📊 Sent={self.frames_sent_total} RGB={self.frames_rgb_total} ~{fps:.1f} FPS (a Jetson)")
                self.last_stat_time = now

        except Exception as e:
            # Estado inválido del REQ (deadlock/timeout/etc.). Re-creamos socket.
            print(f"[MAC] ❌ Error al enviar #{self.frames_sent_total + 1}: {e} → recreando socket")
            try:
                self.sender.close()
            except Exception:
                pass
            time.sleep(0.05)
            self.sender = self._make_sender()
            time.sleep(0.05)

    def on_streaming_client_failure(self, reason, message: str):
        print(f"[MAC] ❌ Streaming failure: {reason} | {message}")

    # ========================= Ciclo principal =========================
    def start(self):
        print(f"[MAC] 🎯 Iniciando conexión con Aria y prueba ZMQ a {self.endpoint}")

        # 1) Prueba rápida ZMQ antes de tocar Aria
        test = np.zeros((120, 160, 3), np.uint8)
        test[:] = (0, 255, 0)
        ok, jpg = cv2.imencode(".jpg", test)
        if ok:
            try:
                r = self.sender.send_jpg("test", jpg)
                print(f"[MAC] ✅ ZMQ OK (reply={r})")
            except Exception as e:
                print(f"[MAC] ❌ ZMQ TEST falló: {e}")
                print("[MAC] 💡 ¿Está el servidor del Jetson escuchando en ese puerto?")
                return

        # 2) Conectar/arrancar Aria
        print("[MAC] 🥽 Conectando al dispositivo Aria…")
        self.device_manager.connect()

        print("[MAC] 📹 Iniciando streaming…")
        _ = self.device_manager.start_streaming()

        print("[MAC] 👁️ Registrando observer…")
        self.device_manager.register_observer(self)

        print("[MAC] 📺 Suscribiendo…")
        self.device_manager.subscribe()

        print("[MAC] ✅ Listo. Enviando frames a Jetson. Ctrl+C para salir.")
        ctrl = CtrlCHandler()
        while not ctrl.should_stop:
            time.sleep(0.5)  # el flujo llega por callback

        self.cleanup()

    def cleanup(self):
        print("\n[MAC] 🧹 Limpieza…")
        try:
            self.device_manager.cleanup()
        except Exception as e:
            print(f"[MAC] ⚠️ Error limpiando DeviceManager: {e}")
        try:
            if hasattr(self, "sender") and self.sender:
                self.sender.close()
        except Exception as e:
            print(f"[MAC] ⚠️ Error cerrando ImageSender: {e}")

        print("[MAC] 📈 Stats finales:",
              f"recibidos={self.frames_recv_total}",
              f"rgb={self.frames_rgb_total}",
              f"enviados={self.frames_sent_total}")

def parse_args():
    ap = argparse.ArgumentParser(description="Mac → Jetson sender (ImageZMQ, robusto)")
    ap.add_argument("--jetson-ip", default="jetson-desktop.local", help="IP/hostname del Jetson")
    ap.add_argument("--port", type=int, default=5555, help="Puerto del Jetson")
    ap.add_argument("--fps", type=int, default=30, help="FPS objetivo desde Aria")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("=" * 60)
    print("MAC CLIENT (ImageZMQ robust)".center(60))
    print(f"Destino: {args.jetson_ip}:{args.port}".center(60))
    print("=" * 60)
    s = MacSender(jetson_ip=args.jetson_ip, port=args.port, fps=args.fps)
    s.start()