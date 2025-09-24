#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 Presentation Manager - Capa UI Separada
Maneja toda la parte de UI/visualización que antes estaba en el Observer

Responsabilidades:
- Gestión de dashboards (OpenCV, Rerun)
- Display de frames procesados
- Logging de eventos y métricas
- UI controls y keyboard input
- Coordinación entre datos y visualización

Fecha: Septiembre 2025
Versión: 1.0 - UI Layer Separation
"""

import cv2
import time
import threading
import numpy as np
import webbrowser
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class UIState:
    """Estado de la interfaz de usuario"""
    dashboard_enabled: bool = False
    display_enabled: bool = True
    window_name: str = "Aria Navigation - TFM"
    window_size: tuple = (800, 600)
    show_fps: bool = True
    show_stats: bool = True


class PresentationManager:
    """
    🎨 Gestor de presentación y UI
    
    Esta clase maneja TODA la parte visual/UI que antes estaba mezclada
    en el Observer original:
    - Dashboard management
    - OpenCV display windows
    - UI controls y keyboard input
    - Logging visual de eventos
    - Coordinación entre datos del sistema y visualización
    """
    
    def __init__(self, enable_dashboard: bool = False, dashboard_type: str = "opencv"):
        """
        Inicializar el gestor de presentación
        
        Args:
            enable_dashboard: Habilitar dashboard
            dashboard_type: Tipo de dashboard ("opencv", "rerun", "web")
        """
        self.ui_state = UIState(dashboard_enabled=enable_dashboard)
        self.dashboard_type = dashboard_type
        self.dashboard = None
        self.dashboard_server_thread = None

        # Estado de visualización
        self.current_display_frame = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # Threading para UI
        self._ui_lock = threading.Lock()
        self._stop_ui = False
        
        # Estadísticas para mostrar
        self.stats_to_display = {}
        
        print(f"🎨 PresentationManager inicializado")
        print(f"  - Dashboard: {'✅' if enable_dashboard else '❌'} ({dashboard_type})")
        
        # Inicializar dashboard si está habilitado
        if enable_dashboard:
            self._initialize_dashboard(dashboard_type)
        
        # Inicializar ventana OpenCV si no hay dashboard
        if not enable_dashboard:
            self._initialize_opencv_window()
    
    def _initialize_dashboard(self, dashboard_type: str):
        """
        Inicializar dashboard según el tipo
        
        Args:
            dashboard_type: Tipo de dashboard a inicializar
        """
        try:
            if dashboard_type == "opencv":
                from presentation.dashboards.opencv_dashboard import OpenCVDashboard
                self.dashboard = OpenCVDashboard()
                print("  ✅ OpenCV Dashboard inicializado")
                
            elif dashboard_type == "rerun":
                from presentation.dashboards.rerun_dashboard import RerunDashboard
                self.dashboard = RerunDashboard()
                print("  ✅ Rerun Dashboard inicializado")

            elif dashboard_type == "web":
                from presentation.dashboards.web_dashboard import WebDashboard
                self.dashboard = WebDashboard()
                try:
                    self.dashboard_server_thread = self.dashboard.start_server()
                    url = f"http://localhost:{self.dashboard.port}"
                    webbrowser.open(url, new=2)
                    print(f"  🌐 Web dashboard abierto en navegador: {url}")
                except Exception as server_err:
                    print(f"  ⚠️ Web dashboard server failed: {server_err}")
                    self.dashboard = None
                    self.ui_state.dashboard_enabled = False
                    return
                print("  ✅ Web Dashboard inicializado")       
            else:
                print(f"  ⚠️ Dashboard type '{dashboard_type}' no reconocido")
                self.ui_state.dashboard_enabled = False
                
        except ImportError as e:
            print(f"  ⚠️ No se pudo importar dashboard {dashboard_type}: {e}")
            self.ui_state.dashboard_enabled = False
        except Exception as e:
            print(f"  ⚠️ Error inicializando dashboard: {e}")
            self.ui_state.dashboard_enabled = False
    
    def _initialize_opencv_window(self):
        """Inicializar ventana OpenCV para display simple"""
        if self.ui_state.display_enabled:
            cv2.namedWindow(self.ui_state.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.ui_state.window_name, *self.ui_state.window_size)
            print(f"  ✅ OpenCV window '{self.ui_state.window_name}' inicializada")
    
    def update_display(self, frame, detections: List[Dict] = None, 
                      motion_state: str = "unknown", 
                      coordinator_stats: Dict = None,
                      depth_map: Optional[np.ndarray] = None,
                      slam1_frame: Optional[np.ndarray] = None,
                      slam2_frame: Optional[np.ndarray] = None,
                      slam_events: Optional[Dict[str, List[Dict]]] = None) -> str:
        """
        🖼️ Actualizar display principal con frame y datos
        
        Args:
            frame: Frame procesado para mostrar
            detections: Detecciones actuales
            motion_state: Estado de movimiento
            coordinator_stats: Estadísticas del coordinator
            depth_map: Último depth map estimado (para dashboards compatibles)
            slam1_frame: Frame más reciente de la cámara SLAM izquierda
            slam2_frame: Frame más reciente de la cámara SLAM derecha
            
        Returns:
            str: Tecla presionada ('q' para quit, 't' para test, etc.)
        """
        if frame is None:
            return ''
        
        # Actualizar FPS
        self._update_fps_counter()
        
        # Almacenar frame para display
        with self._ui_lock:
            self.current_display_frame = frame
        
        # Actualizar estadísticas para mostrar
        self._update_display_stats(detections, motion_state, coordinator_stats)
        
        key_pressed = ''
        
        # Actualizar dashboard si está habilitado
        if self.ui_state.dashboard_enabled and self.dashboard:
            try:
                key_pressed = self._update_dashboard(
                    frame,
                    detections,
                    motion_state,
                    coordinator_stats=coordinator_stats,
                    depth_map=depth_map,
                    slam1_frame=slam1_frame,
                    slam2_frame=slam2_frame,
                    slam_events=slam_events
                )
            except Exception as e:
                print(f"[WARN] Dashboard update failed: {e}")
        
        # Display OpenCV si no hay dashboard
        elif self.ui_state.display_enabled:
            try:
                key_pressed = self._update_opencv_display(frame)
            except Exception as e:
                print(f"[WARN] OpenCV display failed: {e}")
        
        return key_pressed
    
    def _update_dashboard(self, frame, detections, motion_state,
                         coordinator_stats: Optional[Dict] = None,
                         depth_map: Optional[np.ndarray] = None,
                         slam1_frame: Optional[np.ndarray] = None,
                         slam2_frame: Optional[np.ndarray] = None,
                         slam_events: Optional[Dict[str, List[Dict]]] = None) -> str:
        """
        Actualizar dashboard con todos los datos
        
        Returns:
            str: Tecla presionada
        """
        # Log frame principal
        if hasattr(self.dashboard, 'log_rgb_frame'):
            self.dashboard.log_rgb_frame(frame)

        # Log SLAM frames
        slam_events = slam_events or {}
        if slam1_frame is not None and hasattr(self.dashboard, 'log_slam1_frame'):
            self.dashboard.log_slam1_frame(slam1_frame, slam_events.get('slam1'))
        if slam2_frame is not None and hasattr(self.dashboard, 'log_slam2_frame'):
            self.dashboard.log_slam2_frame(slam2_frame, slam_events.get('slam2'))

        # Log detecciones
        if detections and hasattr(self.dashboard, 'log_detections'):
            frame_shape = frame.shape if frame is not None else None
            self.dashboard.log_detections(detections, frame_shape=frame_shape)

        # Log depth map
        if depth_map is not None and hasattr(self.dashboard, 'log_depth_map'):
            self.dashboard.log_depth_map(depth_map)
        
        # Log motion state
        if hasattr(self.dashboard, 'log_motion_state'):
            magnitude = 9.8  # Default, idealmente viene de los datos reales
            self.dashboard.log_motion_state(motion_state, magnitude)
        
        # Log performance metrics
        if hasattr(self.dashboard, 'log_performance_metrics'):
            self.dashboard.log_performance_metrics()
        elif hasattr(self.dashboard, 'update_performance_stats'):
            frames_processed = 0
            if coordinator_stats:
                frames_processed = coordinator_stats.get('frames_processed', 0)
            self.dashboard.update_performance_stats(
                fps=self.current_fps,
                frames_processed=frames_processed
            )
        
        # Log system message con estadísticas
        if hasattr(self.dashboard, 'log_system_message'):
            stats_msg = f"FPS: {self.current_fps:.1f}, Motion: {motion_state}, Detections: {len(detections) if detections else 0}"
            self.dashboard.log_system_message(stats_msg, "STATS")
        
        # Update dashboard y obtener key
        if hasattr(self.dashboard, 'update_all'):
            key = self.dashboard.update_all()
            return chr(key) if key != 255 else ''
        
        return ''
    
    def _update_opencv_display(self, frame) -> str:
        """
        Actualizar display OpenCV simple
        
        Returns:
            str: Tecla presionada
        """
        display_frame = frame

        # Añadir información de overlay directamente sobre el frame mostrado
        if self.ui_state.show_fps:
            cv2.putText(display_frame, f"FPS: {self.current_fps:.1f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if self.ui_state.show_stats and self.stats_to_display:
            y_offset = 60
            for key, value in self.stats_to_display.items():
                text = f"{key}: {value}"
                cv2.putText(display_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
        
        # Mostrar frame
        cv2.imshow(self.ui_state.window_name, display_frame)
        
        # Capturar tecla
        key = cv2.waitKey(1) & 0xFF
        return chr(key) if key != 255 else ''
    
    def _update_fps_counter(self):
        """Actualizar contador de FPS"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.current_fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def _update_display_stats(self, detections, motion_state, coordinator_stats):
        """Actualizar estadísticas para mostrar en UI"""
        self.stats_to_display = {
            'Motion': motion_state,
            'Detections': len(detections) if detections else 0
        }

        if coordinator_stats:
            self.stats_to_display['Frames'] = coordinator_stats.get('frames_processed', 0)
            self.stats_to_display['Audio Queue'] = coordinator_stats.get('audio_queue_size', 0)
            if coordinator_stats.get('slam_events') is not None:
                self.stats_to_display['SLAM Events'] = coordinator_stats.get('slam_events', 0)
    
    def log_system_event(self, message: str, level: str = "INFO"):
        """
        📝 Log evento del sistema (para dashboard)
        
        Args:
            message: Mensaje a loggear
            level: Nivel del mensaje (INFO, WARN, ERROR)
        """
        if self.dashboard and hasattr(self.dashboard, 'log_system_message'):
            try:
                self.dashboard.log_system_message(message, level)
            except Exception as e:
                print(f"[WARN] Dashboard logging failed: {e}")
        else:
            # Fallback a print
            print(f"[{level}] {message}")
    
    def log_audio_command(self, command: str, priority: int = 5):
        """
        🔊 Log comando de audio
        
        Args:
            command: Comando de audio enviado
            priority: Prioridad del comando
        """
        if self.dashboard and hasattr(self.dashboard, 'log_audio_command'):
            try:
                self.dashboard.log_audio_command(command, priority)
            except Exception as e:
                print(f"[WARN] Audio logging failed: {e}")
        
        # También log como evento del sistema
        self.log_system_event(f"AUDIO: {command}", "AUDIO")
    
    def log_detection_event(self, detections: List[Dict]):
        """
        🎯 Log evento de detección importante
        
        Args:
            detections: Lista de detecciones
        """
        if detections:
            for det in detections[:2]:  # Top 2
                name = det.get('name', 'unknown')
                zone = det.get('zone', 'unknown')
                priority = det.get('priority', 0)
                message = f"DETECT: {name} en {zone} (P{priority:.1f})"
                self.log_system_event(message, "DETECT")
    
    def get_current_display_frame(self):
        """Obtener frame actual de display"""
        with self._ui_lock:
            return self.current_display_frame.copy() if self.current_display_frame is not None else None
    
    def get_ui_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de UI"""
        return {
            'dashboard_enabled': self.ui_state.dashboard_enabled,
            'dashboard_type': self.dashboard_type,
            'display_enabled': self.ui_state.display_enabled,
            'current_fps': self.current_fps,
            'frame_count_total': self.frame_count,
            'window_name': self.ui_state.window_name,
            'has_dashboard': self.dashboard is not None
        }
    
    def print_ui_stats(self):
        """Imprimir estadísticas de UI"""
        stats = self.get_ui_stats()
        
        print(f"\n[PRESENTATION STATS]")
        print(f"  FPS: {stats['current_fps']:.1f}")
        print(f"  Dashboard: {'✅' if stats['dashboard_enabled'] else '❌'} ({stats['dashboard_type']})")
        print(f"  Display: {'✅' if stats['display_enabled'] else '❌'}")
        print(f"  Window: {stats['window_name']}")
    
    def cleanup(self):
        """🧹 Limpieza de recursos UI"""
        print("🧹 Limpiando PresentationManager...")
        
        self._stop_ui = True
        
        # Cleanup dashboard
        if self.dashboard:
            try:
                if hasattr(self.dashboard, 'shutdown'):
                    self.dashboard.shutdown()
                elif hasattr(self.dashboard, 'cleanup'):
                    self.dashboard.cleanup()
                print("  ✅ Dashboard cleanup")
            except Exception as e:
                print(f"  ⚠️ Dashboard cleanup error: {e}")
        
        # Cleanup OpenCV
        try:
            cv2.destroyAllWindows()
            print("  ✅ OpenCV windows cleanup")
        except Exception as e:
            print(f"  ⚠️ OpenCV cleanup error: {e}")
        
        print("✅ PresentationManager cleanup completado")


# ============================================================================
# FACTORY FUNCTION PARA CREAR PRESENTATION MANAGER
# ============================================================================

def create_presentation_manager(enable_dashboard: bool = False, 
                              dashboard_type: str = "opencv") -> PresentationManager:
    """
    🏭 Factory function para crear PresentationManager
    
    Args:
        enable_dashboard: Habilitar dashboard
        dashboard_type: Tipo de dashboard ("opencv", "rerun", "web")
        
    Returns:
        PresentationManager: Instancia configurada
    """
    return PresentationManager(enable_dashboard, dashboard_type)


# ============================================================================
# TESTING DEL PRESENTATION MANAGER
# ============================================================================

def test_presentation_manager():
    """Test del PresentationManager"""
    print("🧪 Testing PresentationManager...")
    
    import numpy as np
    
    # Test sin dashboard
    pm = create_presentation_manager(enable_dashboard=False)
    
    # Frame de test
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test detecciones mock
    mock_detections = [
        {'name': 'person', 'zone': 'center', 'priority': 9.5},
        {'name': 'car', 'zone': 'left', 'priority': 7.2}
    ]
    
    # Test coordinator stats mock
    mock_stats = {
        'frames_processed': 150,
        'audio_queue_size': 2
    }
    
    print("  🖼️ Testing display update...")
    key = pm.update_display(test_frame, mock_detections, "walking", mock_stats)
    print(f"  ✅ Display update completed, key: '{key}'")
    
    # Test logging
    print("  📝 Testing event logging...")
    pm.log_system_event("Test system event", "INFO")
    pm.log_audio_command("persona al centro", 9)
    pm.log_detection_event(mock_detections)
    
    # Test stats
    pm.print_ui_stats()
    
    # Test cleanup
    pm.cleanup()
    
    print("✅ PresentationManager test completado!")


if __name__ == "__main__":
    test_presentation_manager()
