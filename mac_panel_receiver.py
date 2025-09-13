#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mac Panel Receiver - Recibe panel de control desde Jetson
TFM - Script para ejecutar EN EL MAC

Propósito: Recibir y mostrar el panel de control que envía el Jetson
con métricas del sistema y estado en tiempo real.

EJECUTAR EN MAC:
python3 mac_panel_receiver.py

Fecha: Día 2+ - Panel remoto de control
Versión: 1.1 - Constructor arreglado
"""

import cv2
import numpy as np
import imagezmq
import time
import signal
from datetime import datetime


class CtrlCHandler:
    """Maneja la señal Ctrl+C para salida limpia."""
    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        print("\n[MAC] Señal de interrupción detectada, cerrando receiver...")
        self.should_stop = True


class PanelReceiver:
    """
    Recibe y procesa los panels enviados desde el Jetson.
    Incluye estadísticas de recepción y control de calidad.
    """
    
    def __init__(self, port=5556):
        self.port = port
        self.frame_count = 0
        self.start_time = None
        self.last_fps_time = time.time()
        self.fps_counter = 0
        self.current_fps = 0
        self.total_bytes_received = 0
        
        # Estadísticas de conexión
        self.connection_stats = {
            'frames_received': 0,
            'frames_dropped': 0,
            'avg_fps': 0,
            'total_mb_received': 0,
            'uptime_seconds': 0
        }
    
    def start_receiving(self, ctrl_handler):
        """
        Inicia la recepción de panels desde el Jetson.
        
        Args:
            ctrl_handler: Handler para control de Ctrl+C
        """
        print(f"[MAC] 📡 Iniciando receiver en puerto {self.port}...")
        
        try:
            # Configurar ImageZMQ Hub (receiver)
            image_hub = imagezmq.ImageHub(open_port=f'tcp://*:{self.port}')
            print(f"[MAC] ✅ Receiver activo en puerto {self.port}")
            print(f"[MAC] 🔗 Esperando conexión desde Jetson...")
            
            # Configurar ventana OpenCV
            window_name = "Jetson Control Panel - Remote View"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 900, 700)
            
            self.start_time = time.time()
            last_stats_time = time.time()
            
            print("[MAC] 📺 Panel receiver activo - Presiona 'q' para salir")
            
            while not ctrl_handler.should_stop:
                try:
                    # Recibir frame del Jetson
                    sender_name, panel_frame = image_hub.recv_image()
                    
                    # Procesar frame recibido
                    processed_frame = self._process_received_frame(panel_frame, sender_name)
                    
                    # Mostrar panel
                    cv2.imshow(window_name, processed_frame)
                    
                    # Enviar confirmación al Jetson
                    image_hub.send_reply(b'PANEL_OK')
                    
                    # Actualizar estadísticas
                    self._update_stats(panel_frame)
                    
                    # Mostrar estadísticas cada 100 frames
                    current_time = time.time()
                    if self.frame_count % 100 == 0:
                        elapsed = current_time - last_stats_time
                        fps = 100 / elapsed if elapsed > 0 else 0
                        print(f"[MAC] 📊 Frames: {self.frame_count}, FPS: {fps:.1f}, "
                              f"MB/s: {(self.total_bytes_received / (1024*1024)) / (current_time - self.start_time):.2f}")
                        last_stats_time = current_time
                    
                    # Verificar teclas de control
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[MAC] 🛑 Tecla 'q' presionada, cerrando receiver...")
                        break
                    elif key == ord('s'):
                        # Guardar screenshot del panel
                        self._save_screenshot(processed_frame)
                    elif key == ord('r'):
                        # Mostrar estadísticas en consola
                        self._print_detailed_stats()
                
                except Exception as e:
                    print(f"[MAC] ❌ Error recibiendo frame: {e}")
                    self.connection_stats['frames_dropped'] += 1
                    continue
            
        except Exception as e:
            print(f"[MAC] ❌ Error en receiver: {e}")
        finally:
            self._cleanup()
    
    def _process_received_frame(self, frame, sender_name):
        """
        Procesa el frame recibido añadiendo información local.
        
        Args:
            frame: Frame recibido del Jetson
            sender_name: Nombre del sender (Jetson)
            
        Returns:
            np.array: Frame procesado con información adicional
        """
        # Crear copia para no modificar el original
        processed = frame.copy()
        
        # Calcular FPS local
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.last_fps_time = current_time
        
        # Añadir overlay con información del Mac
        overlay_height = 80
        overlay = np.zeros((overlay_height, frame.shape[1], 3), dtype=np.uint8)
        overlay[:] = (30, 30, 30)  # Dark overlay
        
        # Información del receiver
        local_time = datetime.now().strftime("%H:%M:%S")
        uptime = current_time - self.start_time if self.start_time else 0
        
        # Textos de overlay
        cv2.putText(overlay, f"MAC RECEIVER | Local Time: {local_time} | Uptime: {uptime:.0f}s", 
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(overlay, f"From: {sender_name} | RX FPS: {self.current_fps} | Total Frames: {self.frame_count}", 
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        
        cv2.putText(overlay, f"Controls: 'q'=Quit | 's'=Screenshot | 'r'=Stats", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        
        # Combinar frame con overlay
        result = np.vstack([processed, overlay])
        
        return result
    
    def _update_stats(self, frame):
        """
        Actualiza las estadísticas de recepción.
        
        Args:
            frame: Frame recibido
        """
        self.frame_count += 1
        self.connection_stats['frames_received'] = self.frame_count
        
        # Calcular tamaño del frame
        frame_size = frame.nbytes
        self.total_bytes_received += frame_size
        self.connection_stats['total_mb_received'] = self.total_bytes_received / (1024 * 1024)
        
        # Uptime
        if self.start_time:
            self.connection_stats['uptime_seconds'] = time.time() - self.start_time
            
            # FPS promedio
            if self.connection_stats['uptime_seconds'] > 0:
                self.connection_stats['avg_fps'] = self.frame_count / self.connection_stats['uptime_seconds']
    
    def _save_screenshot(self, frame):
        """
        Guarda un screenshot del panel actual.
        
        Args:
            frame: Frame a guardar
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"jetson_panel_screenshot_{timestamp}.png"
        
        cv2.imwrite(filename, frame)
        print(f"[MAC] 📸 Screenshot guardado: {filename}")
    
    def _print_detailed_stats(self):
        """
        Muestra estadísticas detalladas en consola.
        """
        print("\n" + "="*50)
        print("📊 ESTADÍSTICAS DETALLADAS DEL RECEIVER")
        print("="*50)
        print(f"🔢 Frames recibidos: {self.connection_stats['frames_received']}")
        print(f"❌ Frames perdidos: {self.connection_stats['frames_dropped']}")
        print(f"📈 FPS promedio: {self.connection_stats['avg_fps']:.2f}")
        print(f"💾 MB recibidos: {self.connection_stats['total_mb_received']:.2f}")
        print(f"⏰ Uptime: {self.connection_stats['uptime_seconds']:.0f} segundos")
        
        # Calcular estadísticas adicionales
        if self.connection_stats['frames_received'] > 0:
            success_rate = (self.connection_stats['frames_received'] / 
                          (self.connection_stats['frames_received'] + self.connection_stats['frames_dropped'])) * 100
            print(f"✅ Tasa de éxito: {success_rate:.1f}%")
            
            avg_frame_size = self.connection_stats['total_mb_received'] / self.connection_stats['frames_received']
            print(f"📏 Tamaño promedio frame: {avg_frame_size*1024:.1f} KB")
        
        print("="*50 + "\n")
    
    def _cleanup(self):
        """
        Limpia recursos al finalizar.
        """
        print("\n[MAC] 🧹 Limpiando recursos...")
        
        # Cerrar ventanas OpenCV
        cv2.destroyAllWindows()
        
        # Mostrar estadísticas finales
        print("\n📊 ESTADÍSTICAS FINALES:")
        print(f"  - Frames totales recibidos: {self.connection_stats['frames_received']}")
        print(f"  - Frames perdidos: {self.connection_stats['frames_dropped']}")
        print(f"  - FPS promedio: {self.connection_stats['avg_fps']:.2f}")
        print(f"  - Total MB recibidos: {self.connection_stats['total_mb_received']:.2f}")
        print(f"  - Tiempo total activo: {self.connection_stats['uptime_seconds']:.0f}s")
        
        if self.connection_stats['frames_received'] > 0:
            success_rate = (self.connection_stats['frames_received'] / 
                          (self.connection_stats['frames_received'] + self.connection_stats['frames_dropped'])) * 100
            print(f"  - Tasa de éxito: {success_rate:.1f}%")
        
        print("[MAC] ✅ Cleanup completado")


def main():
    """
    Función principal del receiver de panel.
    """
    print("=" * 60)
    print("📺 MAC PANEL RECEIVER")
    print("TFM - Recepción de panel de control desde Jetson")
    print("Puerto de escucha: 5556")
    print("=" * 60)
    
    # Handler para salida limpia
    ctrl_handler = CtrlCHandler()
    
    try:
        # Información previa
        print("[MAC] 💡 Preparando receiver...")
        print("[MAC] 🔗 Asegúrate de que el Jetson esté ejecutando jetson_panel_sender.py")
        print("[MAC] ⚠️  El Jetson debe configurar la IP del Mac correctamente")
        print()
        
        # Mostrar IP local para referencia
        import socket
        try:
            # Obtener IP local
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            print(f"[MAC] 📍 IP local detectada: {local_ip}")
            print(f"[MAC] 💡 El Jetson debe usar esta IP en mac_ip = '{local_ip}'")
        except:
            print("[MAC] ⚠️  No se pudo detectar IP local automáticamente")
        
        print()
        input("[MAC] ⏸️  Presiona Enter cuando el Jetson esté listo...")
        
        # Inicializar y ejecutar receiver
        print("[MAC] 🚀 Iniciando panel receiver...")
        panel_receiver = PanelReceiver(port=5556)
        panel_receiver.start_receiving(ctrl_handler)
        
    except KeyboardInterrupt:
        print("\n[MAC] 🛑 Interrupción por teclado detectada")
    except Exception as e:
        print(f"\n[MAC] ❌ Error durante ejecución: {e}")
    finally:
        print("[MAC] 🏁 Panel receiver terminado")


if __name__ == "__main__":
    main()