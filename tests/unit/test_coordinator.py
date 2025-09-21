# tests/unit/test_coordinator_simple.py
"""
🧪 Primer test simple del Coordinator
Guarda este archivo como: tests/unit/test_coordinator_simple.py
"""

import pytest
import numpy as np
from unittest.mock import Mock

# Asegúrate de que el import funciona desde tu estructura
# Ajusta el path según tu estructura de directorios
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.navigation.coordinator import Coordinator


class TestCoordinatorSimple:
    """Tests básicos del Coordinator para empezar"""
    
    def test_coordinator_se_inicializa(self):
        """Test 1: Verificar que el coordinator se puede crear"""
        # Crear mocks simples
        mock_yolo = Mock()
        mock_audio = Mock()
        mock_audio.announcement_cooldown = 2.0
        
        # Crear coordinator
        coordinator = Coordinator(
            yolo_processor=mock_yolo,
            audio_system=mock_audio
        )
        
        # Verificar que se inicializó correctamente
        assert coordinator is not None
        assert coordinator.frames_processed == 0
        assert len(coordinator.current_detections) == 0
        print("✅ Test 1 pasado: Coordinator se inicializa")
    
    def test_process_frame_basico(self):
        """Test 2: Verificar que process_frame funciona"""
        # Setup mocks
        mock_yolo = Mock()
        mock_yolo.process_frame.return_value = [
            {'name': 'person', 'confidence': 0.9, 'bbox': [100, 100, 50, 200]}
        ]
        
        mock_audio = Mock()
        mock_audio.announcement_cooldown = 2.0
        
        coordinator = Coordinator(
            yolo_processor=mock_yolo,
            audio_system=mock_audio
        )
        
        # Crear frame de test
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Procesar frame
        result = coordinator.process_frame(test_frame)
        
        # Verificaciones
        assert isinstance(result, np.ndarray)
        assert result.shape == test_frame.shape
        assert coordinator.frames_processed == 1
        assert len(coordinator.current_detections) == 1
        
        # Verificar que se llamó al YOLO
        mock_yolo.process_frame.assert_called_once()
        print("✅ Test 2 pasado: process_frame funciona")
    
    def test_calculate_zone(self):
        """Test 3: Verificar cálculo de zonas"""
        mock_yolo = Mock()
        mock_audio = Mock()
        mock_audio.announcement_cooldown = 2.0
        
        coordinator = Coordinator(
            yolo_processor=mock_yolo,
            audio_system=mock_audio
        )
        
        # Test zonas
        assert coordinator._calculate_zone(100) == 'left'    # Izquierda
        assert coordinator._calculate_zone(320) == 'center'  # Centro  
        assert coordinator._calculate_zone(500) == 'right'   # Derecha
        
        print("✅ Test 3 pasado: Cálculo de zonas correcto")
    
    def test_estimate_distance(self):
        """Test 4: Verificar estimación de distancia"""
        mock_yolo = Mock()
        mock_audio = Mock()
        mock_audio.announcement_cooldown = 2.0
        
        coordinator = Coordinator(
            yolo_processor=mock_yolo,
            audio_system=mock_audio
        )
        
        # Test distancias para persona
        bbox_cerca = [100, 100, 50, 250]  # height = 250 (cerca)
        bbox_lejos = [100, 100, 50, 80]   # height = 80 (lejos)
        
        assert coordinator._estimate_distance(bbox_cerca, 'person') == 'cerca'
        assert coordinator._estimate_distance(bbox_lejos, 'person') == 'lejos'
        
        print("✅ Test 4 pasado: Estimación de distancia correcta")
    
    def test_get_status(self):
        """Test 5: Verificar que get_status devuelve info correcta"""
        mock_yolo = Mock()
        mock_audio = Mock()
        mock_audio.announcement_cooldown = 2.0
        
        coordinator = Coordinator(
            yolo_processor=mock_yolo,
            audio_system=mock_audio
        )
        
        status = coordinator.get_status()
        
        # Verificar que tiene los campos esperados
        assert 'frames_processed' in status
        assert 'current_detections_count' in status
        assert 'audio_queue_size' in status
        assert 'has_dashboard' in status
        
        print("✅ Test 5 pasado: get_status devuelve datos correctos")


# Test individual para desarrollo rápido
if __name__ == "__main__":
    """
    Modo de desarrollo: puedes correr este archivo directamente
    para hacer debug rápido, pero el testing real se hace con pytest
    """
    print("🧪 Corriendo tests en modo desarrollo...")
    
    test_instance = TestCoordinatorSimple()
    
    try:
        test_instance.test_coordinator_se_inicializa()
        test_instance.test_process_frame_basico() 
        test_instance.test_calculate_zone()
        test_instance.test_estimate_distance()
        test_instance.test_get_status()
        
        print("\n🎉 ¡Todos los tests pasaron en modo desarrollo!")
        print("💡 Para testing profesional, usa: pytest tests/unit/test_coordinator_simple.py")
        
    except Exception as e:
        print(f"\n❌ Test falló: {e}")
        import traceback
        traceback.print_exc()
