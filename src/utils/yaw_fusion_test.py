#!/usr/bin/env python3
"""
Test independiente del sistema de fusión YAW
Conecta con las gafas Aria y prueba el módulo por separado
"""

import time
import math
import aria.sdk as aria
from ..core.device_manager import DeviceManager
from ..motion.yaw_fusion_system import YawFusionSystem, SensorReading, AriaYawIntegrator

class YawFusionTester:
    """
    Tester independiente para el sistema de fusión
    """
    
    def __init__(self):
        self.yaw_integrator = AriaYawIntegrator()
        self.sample_count = 0
        self.last_heading_info = None
        self.start_time = time.time()
        
        # Estadísticas
        self.mag_samples = 0
        self.imu_samples = 0
        self.fusion_updates = 0
        
        print("YawFusionTester inicializado")
    
    def on_magneto_received(self, sample) -> None:
        """Callback del magnetómetro"""
        self.mag_samples += 1
        
        # Pasar al integrador
        self.yaw_integrator.on_magneto_received(sample)
        
        # Intentar obtener heading actualizado
        self._check_heading_update()
    
    def on_imu_received(self, sample) -> None:
        """Callback del IMU"""
        self.imu_samples += 1
        
        # Pasar al integrador
        self.yaw_integrator.on_imu_received(sample)
        
        # Intentar obtener heading actualizado
        self._check_heading_update()
    
    def _check_heading_update(self):
        """Verificar si hay nueva información de heading"""
        heading_info = self.yaw_integrator.get_current_heading()
        
        if heading_info:
            self.last_heading_info = heading_info
            self.fusion_updates += 1
    
    def show_current_status(self):
        """Mostrar estado actual del sistema"""
        elapsed = time.time() - self.start_time
        
        print(f"\n=== ESTADO DEL SISTEMA (t={elapsed:.1f}s) ===")
        print(f"Muestras MAG: {self.mag_samples}")
        print(f"Muestras IMU: {self.imu_samples}")
        print(f"Actualizaciones fusión: {self.fusion_updates}")
        
        if self.last_heading_info:
            print(f"\nÚLTIMO HEADING:")
            print(f"  Ángulo: {self.last_heading_info['heading']:.1f}°")
            print(f"  Dirección: {self.last_heading_info['direction']}")
            print(f"  Calidad: {self.last_heading_info['quality']}")
            print(f"  Estado: {self.last_heading_info['status']}")
            print(f"  Pitch: {self.last_heading_info.get('pitch', 0):.1f}°")
            print(f"  Roll: {self.last_heading_info.get('roll', 0):.1f}°")
        else:
            print("Sin información de heading disponible")
        
        # Estado interno del sistema
        if hasattr(self.yaw_integrator, 'fusion'):
            status = self.yaw_integrator.fusion.get_status_summary()
            print(f"\nESTADO INTERNO:")
            print(f"  Total updates: {status['total_updates']}")
            print(f"  Correcciones MAG: {status['mag_corrections']}")
            print(f"  Tasa corrección: {status['correction_rate']:.2f}")
            print(f"  Bias gyro: {status['gyro_bias']:.4f} rad/s")
    
    def cardinal_direction_test(self):
        """Test de las 4 direcciones cardinales"""
        print(f"\n=== TEST DE DIRECCIONES CARDINALES ===")
        print("Apunta a cada dirección y observa el resultado")
        
        directions = ["NORTE", "ESTE", "SUR", "OESTE"]
        expected_angles = [0, 90, 180, 270]
        results = []
        
        for i, (direction, expected) in enumerate(zip(directions, expected_angles)):
            input(f"\nApunta hacia el {direction} y presiona Enter...")
            
            # Esperar un momento para que se estabilice
            print("Estabilizando... (2 segundos)")
            time.sleep(2)
            
            if self.last_heading_info:
                actual = self.last_heading_info['heading']
                direction_name = self.last_heading_info['direction']
                quality = self.last_heading_info['quality']
                
                # Calcular error
                error = min(
                    abs(actual - expected),
                    abs(actual - expected + 360),
                    abs(actual - expected - 360)
                )
                
                status = "✅" if error < 15 else "⚠️" if error < 30 else "❌"
                
                print(f"Resultado: {actual:.1f}° ({direction_name}) - Error: {error:.1f}° {status}")
                print(f"Calidad: {quality}")
                
                results.append({
                    'direction': direction,
                    'expected': expected,
                    'actual': actual,
                    'error': error,
                    'quality': quality
                })
            else:
                print("❌ No hay datos de heading disponibles")
        
        # Análisis de resultados
        if results:
            print(f"\n=== ANÁLISIS DE RESULTADOS ===")
            total_error = sum(r['error'] for r in results)
            avg_error = total_error / len(results)
            
            print(f"Error promedio: {avg_error:.1f}°")
            
            if avg_error < 10:
                print("🎉 EXCELENTE - Sistema funciona muy bien")
            elif avg_error < 20:
                print("✅ BUENO - Sistema funciona aceptablemente")
            elif avg_error < 40:
                print("⚠️ REGULAR - Funciona pero con limitaciones")
            else:
                print("❌ MALO - Requiere revisión")
            
            # Mostrar detalles
            for r in results:
                print(f"  {r['direction']:>6}: {r['actual']:6.1f}° (error: {r['error']:5.1f}°)")
        
        return results
    
    def continuous_monitoring(self, duration_seconds=30):
        """Monitoreo continuo del sistema"""
        print(f"\n=== MONITOREO CONTINUO ({duration_seconds}s) ===")
        print("Mueve las gafas y observa cómo responde el sistema")
        print("Presiona Ctrl+C para parar antes")
        
        start_time = time.time()
        last_print = 0
        
        try:
            while time.time() - start_time < duration_seconds:
                elapsed = time.time() - start_time
                
                # Mostrar actualización cada 2 segundos
                if elapsed - last_print >= 2.0:
                    if self.last_heading_info:
                        heading = self.last_heading_info['heading']
                        direction = self.last_heading_info['direction']
                        quality = self.last_heading_info['quality']
                        status = self.last_heading_info['status']
                        
                        print(f"t={elapsed:5.1f}s: {heading:6.1f}° ({direction:>8}) [{quality}] ({status})")
                    else:
                        print(f"t={elapsed:5.1f}s: Sin datos")
                    
                    last_print = elapsed
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nMonitoreo interrumpido por el usuario")
        
        print("Monitoreo terminado")

def main():
    print("🧭 YAW FUSION SYSTEM - TEST INDEPENDIENTE")
    print("=" * 60)
    
    try:
        # Crear tester
        tester = YawFusionTester()
        
        # Setup device
        print("Conectando a las gafas Aria...")
        device_manager = DeviceManager()
        device_manager.connect()
        device_manager.start_streaming()
        
        # Observer para el test
        class TestObserver:
            def __init__(self, tester_instance):
                self.tester = tester_instance
            
            def on_magneto_received(self, sample):
                self.tester.on_magneto_received(sample)
            
            def on_imu_received(self, sample):
                self.tester.on_imu_received(sample)
            
            def on_image_received(self, *args):
                pass
            
            def on_baro_received(self, *args):
                pass
            
            def on_streaming_client_failure(self, *args):
                pass
        
        observer = TestObserver(tester)
        device_manager.register_observer(observer)
        device_manager.subscribe()
        
        print("✅ Conectado y recibiendo datos")
        print("Esperando inicialización del sistema...")
        time.sleep(3)
        
        # Menú de opciones
        while True:
            print("\n" + "="*50)
            print("MENÚ DE PRUEBAS")
            print("="*50)
            print("1. 📊 Mostrar estado actual")
            print("2. 🧭 Test direcciones cardinales")
            print("3. 📡 Monitoreo continuo (30s)")
            print("4. 🔧 Monitoreo corto (10s)")
            print("5. 🚪 Salir")
            print("="*50)
            
            choice = input("\nSelecciona opción (1-5): ").strip()
            
            if choice == "1":
                tester.show_current_status()
                
            elif choice == "2":
                results = tester.cardinal_direction_test()
                
                if results:
                    avg_error = sum(r['error'] for r in results) / len(results)
                    if avg_error < 20:
                        print(f"\n🎉 ¡Sistema listo para integrar en tu aplicación principal!")
                        print("El error promedio es aceptable para navegación básica")
                    else:
                        print(f"\n⚠️ Sistema necesita ajustes antes de integración")
                        
            elif choice == "3":
                tester.continuous_monitoring(30)
                
            elif choice == "4":
                tester.continuous_monitoring(10)
                
            elif choice == "5":
                print("Cerrando test...")
                break
                
            else:
                print("❌ Opción inválida")
        
    except KeyboardInterrupt:
        print("\n\nTest interrumpido por el usuario")
        
    except Exception as e:
        print(f"\n❌ Error durante el test: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if 'device_manager' in locals():
            try:
                print("\nLimpiando recursos...")
                device_manager.cleanup()
                print("✅ Cleanup completado")
            except Exception as e:
                print(f"⚠️ Error en cleanup: {e}")
        
        print("\n" + "="*50)
        print("TEST COMPLETADO")
        print("="*50)
        print("Si el sistema funciona bien aquí, está listo para")
        print("integrarse en tu aplicación principal de navegación.")

if __name__ == "__main__":
    main()