import cv2
import numpy as np
import math

class OrientationPanel:
    """Panel visual pequeño para mostrar orientación actual con flecha"""
    
    def __init__(self, size=120):
        self.size = size
        self.center = size // 2
        self.radius = size // 2 - 10
        
    def draw_compass_panel(self, frame, motion_processor, position=(10, 10)):
        """
        Dibujar panel de brújula en el frame
        
        Args:
            frame: Frame OpenCV donde dibujar
            motion_processor: Instance del MotionProcessor
            position: (x, y) posición del panel en el frame
        """
        x, y = position
        
        # Crear área del panel
        panel = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        panel.fill(40)  # Fondo gris oscuro
        
        # Dibujar círculo base
        cv2.circle(panel, (self.center, self.center), self.radius, (100, 100, 100), 2)
        
        # ===== DATOS DE ORIENTACIÓN =====
        try:
            # Obtener orientación del motion processor
            yaw_degrees = motion_processor.get_yaw_degrees()
            mag_heading = motion_processor.get_absolute_heading()
            is_moving = motion_processor.orientation.is_moving
            
            # ===== FLECHA PRINCIPAL (YAW GIROSCOPIO) =====
            yaw_rad = math.radians(yaw_degrees - 90)  # -90 para que 0° sea arriba
            end_x = self.center + int(self.radius * 0.8 * math.cos(yaw_rad))
            end_y = self.center + int(self.radius * 0.8 * math.sin(yaw_rad))
            
            # Color según estado de movimiento
            arrow_color = (0, 255, 0) if not is_moving else (0, 255, 255)  # Verde/Amarillo
            
            # Dibujar flecha principal (Gyro)
            cv2.arrowedLine(panel, (self.center, self.center), (end_x, end_y), 
                           arrow_color, 3, tipLength=0.3)
            
            # ===== INDICADOR MAGNETÓMETRO =====
            if mag_heading != 0:
                mag_rad = math.radians(mag_heading - 90)
                mag_end_x = self.center + int(self.radius * 0.6 * math.cos(mag_rad))
                mag_end_y = self.center + int(self.radius * 0.6 * math.sin(mag_rad))
                
                # Línea más fina para magnetómetro
                cv2.line(panel, (self.center, self.center), (mag_end_x, mag_end_y), 
                        (255, 0, 255), 2)  # Magenta
            
            # ===== MARCADORES CARDINALES =====
            cardinal_points = [
                (0, "N", (255, 255, 255)),    # Norte - Blanco
                (90, "E", (150, 150, 150)),   # Este - Gris
                (180, "S", (150, 150, 150)),  # Sur - Gris  
                (270, "W", (150, 150, 150))   # Oeste - Gris
            ]
            
            for angle, label, color in cardinal_points:
                angle_rad = math.radians(angle - 90)
                label_x = self.center + int((self.radius + 15) * math.cos(angle_rad))
                label_y = self.center + int((self.radius + 15) * math.sin(angle_rad))
                cv2.putText(panel, label, (label_x - 5, label_y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # ===== INFORMACIÓN TEXTUAL =====
            # Yaw actual
            cv2.putText(panel, f"Y:{yaw_degrees:.0f}", (5, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, arrow_color, 1)
            
            # Heading magnetómetro
            if mag_heading != 0:
                cv2.putText(panel, f"M:{mag_heading:.0f}", (5, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            
            # Estado de movimiento
            status = "WALK" if is_moving else "STILL"
            status_color = (0, 255, 255) if is_moving else (0, 255, 0)
            cv2.putText(panel, status, (5, self.size - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, status_color, 1)
            
        except Exception as e:
            # En caso de error, mostrar panel básico
            cv2.putText(panel, "NO DATA", (25, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # ===== INSERTAR PANEL EN EL FRAME =====
        # Verificar que el panel cabe en el frame
        frame_h, frame_w = frame.shape[:2]
        if y + self.size <= frame_h and x + self.size <= frame_w:
            # Crear máscara para bordes suavizados
            panel_gray = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(panel_gray, 35, 255, cv2.THRESH_BINARY)
            
            # Aplicar panel al frame
            panel_area = frame[y:y+self.size, x:x+self.size]
            panel_area[mask > 0] = panel[mask > 0]
        
        return frame
    
    def draw_direction_text(self, frame, motion_processor, object_center_x, frame_width):
        """
        Mostrar texto de dirección contextual para testing
        """
        try:
            contextual_dir = motion_processor.get_contextual_direction(object_center_x, frame_width)
            
            # Mostrar en la parte superior del frame
            cv2.putText(frame, f"Objeto: {contextual_dir}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
        except Exception as e:
            cv2.putText(frame, "Direction: ERROR", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame


def add_orientation_overlay(frame, motion_processor, show_panel=True, show_direction=False, object_x=None, frame_w=None):
    """
    Función de utilidad para añadir overlay de orientación completo
    
    Args:
        frame: Frame OpenCV
        motion_processor: MotionProcessor instance
        show_panel: Mostrar panel de brújula
        show_direction: Mostrar texto direccional
        object_x: Posición X de objeto para dirección contextual
        frame_w: Ancho del frame para cálculo direccional
    """
    if show_panel:
        panel = OrientationPanel()
        frame = panel.draw_compass_panel(frame, motion_processor)
    
    if show_direction and object_x is not None and frame_w is not None:
        panel = OrientationPanel()
        frame = panel.draw_direction_text(frame, motion_processor, object_x, frame_w)
    
    return frame