#!/usr/bin/env python3
"""
Ejemplo de uso del MockObserver para desarrollo sin gafas Aria.

Demuestra los 3 modos de operación:
1. Synthetic: Frames generados proceduralmente
2. Video: Replay de video grabado
3. Static: Imagen estática con variaciones
"""

import sys
import time
import cv2
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.mock_observer import MockObserver


def test_synthetic_mode():
    """Test modo sintético (frames generados)."""
    print("\n" + "="*60)
    print("TEST 1: MODO SINTÉTICO")
    print("="*60)
    
    observer = MockObserver(
        mode='synthetic',
        fps=30,
        resolution=(1408, 1408)
    )
    
    observer.start()
    
    # Capturar algunos frames
    for i in range(10):
        time.sleep(0.1)
        frame = observer.get_latest_frame()
        if frame is not None:
            print(f"  Frame {i+1}: shape={frame.shape}, dtype={frame.dtype}")
            
            # Mostrar cada 3 frames (opcional)
            if i % 3 == 0:
                cv2.imshow("MockObserver - Synthetic", cv2.resize(frame, (704, 704)))
                cv2.waitKey(1)
    
    stats = observer.get_stats()
    print(f"\nEstadísticas:")
    print(f"  Frames generados: {stats['frames_generated']}")
    print(f"  FPS actual: {stats['actual_fps']:.1f}")
    print(f"  Buffer size: {stats['buffer_size']}")
    
    observer.stop()
    cv2.destroyAllWindows()
    print("✅ Test sintético completado\n")


def test_video_mode():
    """Test modo video (si existe un video)."""
    print("\n" + "="*60)
    print("TEST 2: MODO VIDEO")
    print("="*60)
    
    # Buscar un video de ejemplo
    video_paths = [
        "data/session.mp4",
        "logs/recording.mp4",
        "../test_video.mp4"
    ]
    
    video_path = None
    for path in video_paths:
        if Path(path).exists():
            video_path = path
            break
    
    if not video_path:
        print("⚠️  No se encontró video de prueba, skip")
        print(f"   Puedes probar con: observer = MockObserver(mode='video', video_path='tu_video.mp4')")
        return
    
    observer = MockObserver(
        mode='video',
        video_path=video_path,
        fps=30
    )
    
    observer.start()
    
    # Capturar frames del video
    for i in range(10):
        time.sleep(0.1)
        frame = observer.get_latest_frame()
        if frame is not None:
            print(f"  Frame {i+1} del video: shape={frame.shape}")
            
            if i % 3 == 0:
                cv2.imshow("MockObserver - Video", cv2.resize(frame, (704, 704)))
                cv2.waitKey(1)
    
    stats = observer.get_stats()
    print(f"\nEstadísticas:")
    print(f"  Frames reproducidos: {stats['frames_generated']}")
    print(f"  FPS actual: {stats['actual_fps']:.1f}")
    
    observer.stop()
    cv2.destroyAllWindows()
    print("✅ Test video completado\n")


def test_static_mode():
    """Test modo estático (imagen con variaciones)."""
    print("\n" + "="*60)
    print("TEST 3: MODO ESTÁTICO")
    print("="*60)
    
    # Buscar una imagen de ejemplo
    image_paths = [
        "data/test_frame.jpg",
        "logs/frame.png",
        "../test_image.jpg"
    ]
    
    image_path = None
    for path in image_paths:
        if Path(path).exists():
            image_path = path
            break
    
    if not image_path:
        print("⚠️  No se encontró imagen de prueba, creando una sintética")
        # Crear imagen de prueba
        import numpy as np
        test_img = np.random.randint(50, 200, (1408, 1408, 3), dtype=np.uint8)
        cv2.rectangle(test_img, (400, 400), (1000, 1000), (100, 150, 200), -1)
        image_path = "/tmp/mock_test_image.jpg"
        cv2.imwrite(image_path, test_img)
        print(f"   Imagen creada en: {image_path}")
    
    observer = MockObserver(
        mode='static',
        image_path=image_path,
        fps=30
    )
    
    observer.start()
    
    # Capturar frames (deberían ser similares pero con pequeñas variaciones)
    for i in range(10):
        time.sleep(0.1)
        frame = observer.get_latest_frame()
        if frame is not None:
            print(f"  Frame {i+1} estático: shape={frame.shape}")
            
            if i % 3 == 0:
                cv2.imshow("MockObserver - Static", cv2.resize(frame, (704, 704)))
                cv2.waitKey(1)
    
    stats = observer.get_stats()
    print(f"\nEstadísticas:")
    print(f"  Frames generados: {stats['frames_generated']}")
    print(f"  FPS actual: {stats['actual_fps']:.1f}")
    
    observer.stop()
    cv2.destroyAllWindows()
    print("✅ Test estático completado\n")


def test_context_manager():
    """Test uso con context manager."""
    print("\n" + "="*60)
    print("TEST 4: CONTEXT MANAGER")
    print("="*60)
    
    with MockObserver(mode='synthetic', fps=60) as observer:
        time.sleep(0.5)
        stats = observer.get_stats()
        print(f"  Context manager funcionando")
        print(f"  Frames generados: {stats['frames_generated']}")
        print(f"  FPS: {stats['actual_fps']:.1f}")
    
    print("✅ Context manager completado (auto cleanup)\n")


if __name__ == "__main__":
    print("\n🧪 Testing MockObserver")
    print("="*60)
    
    try:
        test_synthetic_mode()
        test_video_mode()
        test_static_mode()
        test_context_manager()
        
        print("\n" + "="*60)
        print("✅ TODOS LOS TESTS COMPLETADOS")
        print("="*60)
        print("\nEl MockObserver está listo para usar en main.py")
        print("Puedes desarrollar sin las gafas Aria! 🚀")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrumpidos por usuario")
    except Exception as e:
        print(f"\n❌ Error en tests: {e}")
        import traceback
        traceback.print_exc()
