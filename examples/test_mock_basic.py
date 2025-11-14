#!/usr/bin/env python3
"""
Test básico del MockObserver sin dependencia de OpenCV para display.
Solo valida que el mock funciona correctamente.
"""

import sys
import time
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.mock_observer import MockObserver


def test_mock_basic():
    """Test básico del MockObserver."""
    print("\n" + "="*70)
    print("🧪 TEST BÁSICO: MockObserver")
    print("="*70)
    
    print("\n1️⃣  Inicializando MockObserver en modo sintético...")
    observer = MockObserver(
        mode='synthetic',
        fps=60,
        resolution=(1408, 1408)
    )
    print("   ✅ MockObserver creado")
    
    print("\n2️⃣  Iniciando generación de frames...")
    observer.start()
    print("   ✅ Generación iniciada")
    
    print("\n3️⃣  Esperando frames (2 segundos)...")
    time.sleep(2.0)
    
    print("\n4️⃣  Capturando frames...")
    frames_captured = 0
    for i in range(10):
        frame = observer.get_latest_frame()
        if frame is not None:
            frames_captured += 1
            if i == 0:  # Solo mostrar detalles del primer frame
                print(f"   📸 Frame capturado:")
                print(f"      - Shape: {frame.shape}")
                print(f"      - Dtype: {frame.dtype}")
                print(f"      - Min value: {frame.min()}")
                print(f"      - Max value: {frame.max()}")
        time.sleep(0.1)
    
    print(f"\n   ✅ Capturados {frames_captured}/10 frames")
    
    print("\n5️⃣  Obteniendo estadísticas...")
    stats = observer.get_stats()
    print(f"   📊 Estadísticas del MockObserver:")
    print(f"      - Modo: {stats['mode']}")
    print(f"      - Frames generados: {stats['frames_generated']}")
    print(f"      - Tiempo transcurrido: {stats['elapsed_time']:.2f}s")
    print(f"      - FPS objetivo: {stats['target_fps']}")
    print(f"      - FPS real: {stats['actual_fps']:.1f}")
    print(f"      - Buffer size: {stats['buffer_size']}")
    print(f"      - Running: {stats['running']}")
    
    print("\n6️⃣  Deteniendo MockObserver...")
    observer.stop()
    print("   ✅ MockObserver detenido")
    
    # Validaciones
    print("\n7️⃣  Validando resultados...")
    assert frames_captured > 5, "Deben capturarse al menos 5 frames"
    assert stats['frames_generated'] > 100, "Deben generarse > 100 frames en 2s @ 60fps"
    assert stats['actual_fps'] > 30, f"FPS debe ser > 30, obtenido: {stats['actual_fps']:.1f}"
    print("   ✅ Todas las validaciones pasaron")
    
    return True


def test_context_manager():
    """Test del context manager."""
    print("\n" + "="*70)
    print("🧪 TEST CONTEXT MANAGER")
    print("="*70)
    
    print("\n1️⃣  Usando MockObserver con 'with' statement...")
    with MockObserver(mode='synthetic', fps=60) as observer:
        print("   ✅ MockObserver iniciado automáticamente")
        time.sleep(1.0)
        
        frame = observer.get_latest_frame()
        assert frame is not None, "Frame debe estar disponible"
        print(f"   📸 Frame obtenido: shape={frame.shape}")
        
        stats = observer.get_stats()
        print(f"   📊 FPS: {stats['actual_fps']:.1f}")
    
    print("   ✅ MockObserver detenido automáticamente (context manager)")
    return True


def test_api_compatibility():
    """Test que la API es compatible con Observer real."""
    print("\n" + "="*70)
    print("🧪 TEST API COMPATIBILITY")
    print("="*70)
    
    observer = MockObserver(mode='synthetic')
    observer.start()
    time.sleep(0.5)
    
    # Test métodos que debe tener para ser compatible
    print("\n   Verificando métodos de la API...")
    
    # get_latest_frame()
    frame = observer.get_latest_frame()
    assert frame is not None
    print("   ✅ get_latest_frame() - OK")
    
    # get_frame_data()
    frame_data = observer.get_frame_data()
    assert frame_data is not None
    assert 'frame' in frame_data
    assert 'timestamp' in frame_data
    assert 'frame_id' in frame_data
    print("   ✅ get_frame_data() - OK")
    
    # get_buffer_size()
    buffer_size = observer.get_buffer_size()
    assert buffer_size > 0
    print(f"   ✅ get_buffer_size() = {buffer_size} - OK")
    
    # get_stats()
    stats = observer.get_stats()
    assert 'mode' in stats
    assert 'frames_generated' in stats
    print("   ✅ get_stats() - OK")
    
    observer.stop()
    print("\n   ✅ API completamente compatible con Observer real")
    return True


if __name__ == "__main__":
    print("\n🚀 TESTING MOCKOBSERVER")
    print("="*70)
    print("Esto permite desarrollar sin las gafas Aria")
    print("="*70)
    
    success = True
    
    try:
        # Test 1: Funcionalidad básica
        if not test_mock_basic():
            success = False
        
        # Test 2: Context manager
        if not test_context_manager():
            success = False
        
        # Test 3: API compatibility
        if not test_api_compatibility():
            success = False
        
        if success:
            print("\n" + "="*70)
            print("✅ TODOS LOS TESTS PASARON")
            print("="*70)
            print("\n🎉 El MockObserver está listo para usar!")
            print("\n📝 Próximos pasos:")
            print("   1. Ejecuta main.py y selecciona modo Mock (opción 2)")
            print("   2. Desarrolla las optimizaciones FASE 1")
            print("   3. Crea benchmarks sintéticos")
            print("   4. Cuando tengas las gafas, testea con modo real (opción 1)")
            print("\n💡 Ejemplo de uso en código:")
            print("   from core.mock_observer import MockObserver")
            print("   observer = MockObserver(mode='synthetic', fps=60)")
            print("   observer.start()")
            print("   frame = observer.get_latest_frame()")
            print("   observer.stop()")
        else:
            print("\n❌ ALGUNOS TESTS FALLARON")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrumpidos por usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR EN TESTS: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
