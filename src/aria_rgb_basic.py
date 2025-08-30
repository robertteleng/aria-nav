#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de navegación para personas ciegas usando gafas Meta Aria
TFM - Día 1: Stream RGB básico con observer personalizado

Fecha: 30/08/2025
Versión: 1.0 - RGB streaming básico
"""

import signal
import cv2
import numpy as np
import aria.sdk as aria


class CtrlCHandler:
    """
    Maneja la señal Ctrl+C para salida limpia del programa.
    Evita corrupción de datos y desconexión abrupta del dispositivo.
    """
    def __init__(self):
        self.should_stop = False
        # Registrar handler para SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Callback ejecutado cuando se detecta Ctrl+C"""
        print("\n[INFO] Señal de interrupción detectada, cerrando limpiamente...")
        self.should_stop = True


class AriaRgbObserver:
    """
    Observer personalizado para recibir solo stream RGB de las gafas Aria.
    
    Implementa el patrón Observer para callbacks asíncronos del SDK de Aria.
    Filtra solo imágenes RGB y aplica transformaciones necesarias.
    """
    
    def __init__(self):
        # Frame más reciente recibido
        self.current_frame = None
        # Contador para estadísticas
        self.frame_count = 0
    
    def on_image_received(self, image: np.array, record) -> None:
        """
        Callback invocado por el SDK cuando llega una nueva imagen.
        
        Args:
            image: Array NumPy con los datos de la imagen (BGR)
            record: Metadata de la imagen (timestamp, camera_id, etc.)
        """
        # Filtrar solo cámara RGB (ignorar SLAM1, SLAM2, EyeTrack)
        if record.camera_id == aria.CameraId.Rgb:
            # Rotar imagen 90° para orientación correcta
            # Las cámaras Aria están montadas lateralmente
            rotated_image = np.rot90(image)
            
            # Actualizar frame actual (thread-safe simple)
            self.current_frame = rotated_image
            self.frame_count += 1
            
            # Debug: mostrar estadísticas cada 100 frames
            if self.frame_count % 100 == 0:
                print(f"[DEBUG] Frames RGB procesados: {self.frame_count}")
    
    def get_latest_frame(self):
        """
        Obtiene el frame más reciente disponible.
        
        Returns:
            np.array or None: Último frame RGB rotado, o None si no hay datos
        """
        return self.current_frame


def connect_aria_device():
    """
    Establece conexión con el dispositivo Aria.
    
    Returns:
        tuple: (device_client, device) - Cliente y dispositivo conectado
        
    Raises:
        Exception: Si no se puede conectar al dispositivo
    """
    print("[INFO] Iniciando conexión con gafas Aria...")
    
    # Crear cliente del dispositivo
    device_client = aria.DeviceClient()
    
    # TODO: Añadir configuración para IP específica si se usa WiFi
    # client_config = aria.DeviceClientConfig()
    # client_config.ip_v4_address = "192.168.1.100"
    # device_client.set_client_config(client_config)
    
    # Conectar al dispositivo (por defecto USB)
    device = device_client.connect()
    
    print("[INFO] ✓ Conexión establecida exitosamente")
    return device_client, device


def setup_rgb_streaming(device):
    """
    Configura el streaming para capturar solo datos de imagen.
    
    Args:
        device: Dispositivo Aria conectado
        
    Returns:
        StreamingManager: Manager configurado y streaming iniciado
    """
    print("[INFO] Configurando streaming RGB...")
    
    # Obtener el manager de streaming del dispositivo
    streaming_manager = device.streaming_manager
    
    # Crear configuración de streaming
    streaming_config = aria.StreamingConfig()
    
    # Usar perfil predefinido (profile18 es estándar para RGB)
    streaming_config.profile_name = "profile18"
    
    # Configurar interfaz USB (más estable que WiFi para desarrollo)
    streaming_config.streaming_interface = aria.StreamingInterface.Usb
    
    # Usar certificados efímeros (no requiere setup manual de certificados)
    streaming_config.security_options.use_ephemeral_certs = True
    
    # Aplicar configuración al manager
    streaming_manager.streaming_config = streaming_config
    
    # Iniciar el streaming en el dispositivo
    streaming_manager.start_streaming()
    
    print("[INFO] ✓ Streaming RGB iniciado")
    return streaming_manager


def main():
    """
    Función principal del programa.
    Orquesta la conexión, streaming y visualización.
    """
    print("=" * 60)
    print("TFM - Sistema de navegación para ciegos")
    print("Día 1: Stream RGB básico desde gafas Aria")
    print("=" * 60)
    
    # Configurar handler para salida limpia
    ctrl_handler = CtrlCHandler()
    
    # Variables para cleanup en caso de error
    device_client = None
    streaming_manager = None
    streaming_client = None
    
    try:
        # 1. CONEXIÓN AL DISPOSITIVO
        device_client, device = connect_aria_device()
        
        # 2. CONFIGURACIÓN DE STREAMING
        streaming_manager = setup_rgb_streaming(device)
        
        # 3. SETUP DEL OBSERVER
        print("[INFO] Configurando observer RGB...")
        
        # Crear nuestro observer personalizado
        rgb_observer = AriaRgbObserver()
        
        # Obtener cliente de streaming
        streaming_client = streaming_manager.streaming_client
        
        # Registrar nuestro observer
        streaming_client.set_streaming_client_observer(rgb_observer)
        
        # TODO: Configurar filtro para solo imágenes (opcional)
        # subscription_config = streaming_client.subscription_config
        # subscription_config.subscriber_data_type = aria.StreamingDataType.Image
        
        # Iniciar suscripción al stream
        streaming_client.subscribe()
        print("[INFO] ✓ Suscripción al stream RGB activa")
        
        # 4. CONFIGURACIÓN DE VISUALIZACIÓN
        window_name = "Aria RGB Stream - TFM Navegación"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        print("[INFO] Stream activo - Presiona 'q' para salir o Ctrl+C")
        print("[INFO] Esperando frames RGB...")
        
        # 5. LOOP PRINCIPAL DE VISUALIZACIÓN
        frames_displayed = 0
        
        while not ctrl_handler.should_stop:
            # Obtener frame más reciente
            current_frame = rgb_observer.get_latest_frame()
            
            # Si hay frame disponible, mostrarlo
            if current_frame is not None:
                cv2.imshow(window_name, current_frame)
                frames_displayed += 1
                
                # Estadísticas cada 200 frames mostrados
                if frames_displayed % 200 == 0:
                    print(f"[INFO] Frames mostrados: {frames_displayed}")
            
            # Verificar si se presionó 'q' para salir
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Tecla 'q' detectada, cerrando aplicación...")
                break
        
        # Mostrar estadísticas finales
        print(f"[INFO] Estadísticas finales:")
        print(f"  - Frames RGB recibidos: {rgb_observer.frame_count}")
        print(f"  - Frames mostrados: {frames_displayed}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Interrupción por teclado detectada")
        
    except Exception as e:
        print(f"[ERROR] Error durante ejecución: {e}")
        print("[ERROR] Revisa conexión del dispositivo y dependencias")
        
    finally:
        # 6. CLEANUP ORDENADO DE RECURSOS
        print("[INFO] Iniciando limpieza de recursos...")
        
        try:
            # Desuscribirse del stream
            if streaming_client:
                streaming_client.unsubscribe()
                print("[INFO] ✓ Desuscripción completada")
        except Exception as e:
            print(f"[WARN] Error en unsubscribe: {e}")
        
        try:
            # Detener streaming
            if streaming_manager:
                streaming_manager.stop_streaming()
                print("[INFO] ✓ Streaming detenido")
        except Exception as e:
            print(f"[WARN] Error deteniendo streaming: {e}")
        
        try:
            # Desconectar dispositivo
            if device_client and 'device' in locals():
                device_client.disconnect(device)
                print("[INFO] ✓ Dispositivo desconectado")
        except Exception as e:
            print(f"[WARN] Error en desconexión: {e}")
        
        # Cerrar ventanas OpenCV
        cv2.destroyAllWindows()
        print("[INFO] ✓ Ventanas cerradas")
        
        print("[INFO] Programa terminado exitosamente")


if __name__ == "__main__":
    # Ejecutar programa principal
    main()