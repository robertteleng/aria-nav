#!/usr/bin/env python3
"""
Test completamente exhaustivo del magnetómetro
Incluye TODAS las combinaciones posibles, incluyendo las que faltaban
"""

import math
import time
from ..core.device_manager import DeviceManager
import numpy as np

class CompleteExhaustiveMagnetometerTest:
    def __init__(self):
        self.last_mag_data = None
        self.sample_count = 0
        self.imu_count = 0
        
        self.last_accel_data = None
        self.DECLINATION_DEG = -0.4  # declinación magnética aproximada para Alicante
        
        # TODAS las combinaciones posibles (60+ combinaciones)
        self.all_combinations = [
            # === BÁSICAS X,Y ===
            ("X,Y", lambda x, y, z: math.atan2(y, x)),
            ("Y,X", lambda x, y, z: math.atan2(x, y)),
            
            # === CON NEGATIVOS X,Y ===
            ("-X,Y", lambda x, y, z: math.atan2(y, -x)),
            ("X,-Y", lambda x, y, z: math.atan2(-y, x)),
            ("-X,-Y", lambda x, y, z: math.atan2(-y, -x)),
            ("-Y,X", lambda x, y, z: math.atan2(x, -y)),
            ("Y,-X", lambda x, y, z: math.atan2(-x, y)),
            ("-Y,-X", lambda x, y, z: math.atan2(-x, -y)),
            
            # === X CON Z ===
            ("X,Z", lambda x, y, z: math.atan2(z, x)),
            ("Z,X", lambda x, y, z: math.atan2(x, z)),
            ("-X,Z", lambda x, y, z: math.atan2(z, -x)),
            ("X,-Z", lambda x, y, z: math.atan2(-z, x)),
            ("-X,-Z", lambda x, y, z: math.atan2(-z, -x)),
            ("Z,-X", lambda x, y, z: math.atan2(-x, z)),
            ("-Z,X", lambda x, y, z: math.atan2(x, -z)),
            ("-Z,-X", lambda x, y, z: math.atan2(-x, -z)),
            
            # === Y CON Z ===
            ("Y,Z", lambda x, y, z: math.atan2(z, y)),
            ("Z,Y", lambda x, y, z: math.atan2(y, z)),
            ("-Y,Z", lambda x, y, z: math.atan2(z, -y)),
            ("Y,-Z", lambda x, y, z: math.atan2(-z, y)),
            ("-Y,-Z", lambda x, y, z: math.atan2(-z, -y)),
            ("Z,-Y", lambda x, y, z: math.atan2(-y, z)),
            ("-Z,Y", lambda x, y, z: math.atan2(y, -z)),
            ("-Z,-Y", lambda x, y, z: math.atan2(-y, -z)),
            
            # === INTERCAMBIOS DE EJES (que faltaban) ===
            # X -> Y, Y -> Z, Z -> X (rotación de ejes)
            ("Y_as_X,Z_as_Y", lambda x, y, z: math.atan2(z, y)),
            ("Z_as_X,X_as_Y", lambda x, y, z: math.atan2(x, z)),
            ("X_as_X,Y_as_Y", lambda x, y, z: math.atan2(y, x)),  # Standard
            
            # Con negativos en intercambios
            ("-Y_as_X,Z_as_Y", lambda x, y, z: math.atan2(z, -y)),
            ("Y_as_X,-Z_as_Y", lambda x, y, z: math.atan2(-z, y)),
            ("-Y_as_X,-Z_as_Y", lambda x, y, z: math.atan2(-z, -y)),
            
            ("-Z_as_X,X_as_Y", lambda x, y, z: math.atan2(x, -z)),
            ("Z_as_X,-X_as_Y", lambda x, y, z: math.atan2(-x, z)),
            ("-Z_as_X,-X_as_Y", lambda x, y, z: math.atan2(-x, -z)),
            
            # === TRANSFORMACIONES ANGULARES ===
            ("X,Y+90°", lambda x, y, z: math.atan2(y, x) + math.pi/2),
            ("X,Y-90°", lambda x, y, z: math.atan2(y, x) - math.pi/2),
            ("X,Y+180°", lambda x, y, z: math.atan2(y, x) + math.pi),
            ("X,Y+270°", lambda x, y, z: math.atan2(y, x) + 3*math.pi/2),
            
            ("-X,Y+90°", lambda x, y, z: math.atan2(y, -x) + math.pi/2),
            ("-X,Y-90°", lambda x, y, z: math.atan2(y, -x) - math.pi/2),
            ("X,-Y+90°", lambda x, y, z: math.atan2(-y, x) + math.pi/2),
            ("X,-Y-90°", lambda x, y, z: math.atan2(-y, x) - math.pi/2),
            
            # === COMBINACIONES CON Z TRANSFORMADAS ===
            ("Z,Y+90°", lambda x, y, z: math.atan2(y, z) + math.pi/2),
            ("Z,Y-90°", lambda x, y, z: math.atan2(y, z) - math.pi/2),
            ("Z,Y+180°", lambda x, y, z: math.atan2(y, z) + math.pi),
            
            ("-Z,Y+90°", lambda x, y, z: math.atan2(y, -z) + math.pi/2),
            ("-Z,Y-90°", lambda x, y, z: math.atan2(y, -z) - math.pi/2),
            ("Z,-Y+90°", lambda x, y, z: math.atan2(-y, z) + math.pi/2),
            ("Z,-Y-90°", lambda x, y, z: math.atan2(-y, z) - math.pi/2),
            
            # === COMBINACIONES CON MAGNITUDES ===
            ("mag(X),Y", lambda x, y, z: math.atan2(y, abs(x))),
            ("X,mag(Y)", lambda x, y, z: math.atan2(abs(y), x)),
            ("mag(X),mag(Y)", lambda x, y, z: math.atan2(abs(y), abs(x))),
            
            ("mag(Z),Y", lambda x, y, z: math.atan2(y, abs(z))),
            ("Z,mag(Y)", lambda x, y, z: math.atan2(abs(y), z)),
            ("mag(Z),mag(Y)", lambda x, y, z: math.atan2(abs(y), abs(z))),
            
            # === COMBINACIONES HÍBRIDAS ===
            ("X+Y,Z", lambda x, y, z: math.atan2(z, x+y)),
            ("X-Y,Z", lambda x, y, z: math.atan2(z, x-y)),
            ("Y-X,Z", lambda x, y, z: math.atan2(z, y-x)),
            
            ("X,Y+Z", lambda x, y, z: math.atan2(y+z, x)),
            ("X,Y-Z", lambda x, y, z: math.atan2(y-z, x)),
            ("X,Z-Y", lambda x, y, z: math.atan2(z-y, x)),
            
            # === COMBINACIONES CON RAÍCES (experimentales) ===
            ("sqrt(X²+Y²),Z", lambda x, y, z: math.atan2(z, math.sqrt(x*x + y*y))),
            ("X,sqrt(Y²+Z²)", lambda x, y, z: math.atan2(math.sqrt(y*y + z*z), x)),
            ("sqrt(X²+Z²),Y", lambda x, y, z: math.atan2(y, math.sqrt(x*x + z*z))),
            
            # === INVERSIONES COMPLEJAS ===
            ("1/X,Y", lambda x, y, z: math.atan2(y, 1/x) if abs(x) > 0.001 else None),
            ("X,1/Y", lambda x, y, z: math.atan2(1/y, x) if abs(y) > 0.001 else None),
            ("1/Z,Y", lambda x, y, z: math.atan2(y, 1/z) if abs(z) > 0.001 else None),
            
            # === FUNCIONES TRIGONOMÉTRICAS ===
            ("sin(X),cos(Y)", lambda x, y, z: math.atan2(math.cos(y), math.sin(x))),
            ("cos(X),sin(Y)", lambda x, y, z: math.atan2(math.sin(y), math.cos(x))),
            ("tan(X),Y", lambda x, y, z: math.atan2(y, math.tan(x)) if abs(math.cos(x)) > 0.001 else None),
        ]
        
        print(f"🔍 Complete Exhaustive Test inicializado")
        print(f"   Combinaciones totales: {len(self.all_combinations)}")
    
    def on_magneto_received(self, *args) -> None:
        """Callback del magnetómetro (firma flexible)."""
        if not args:
            return

        mag_obj = None
        xyz = None

        for a in reversed(args):
            # Objeto con .mag_tesla
            if hasattr(a, "mag_tesla"):
                mag_obj = a
                break
            # dict con mx,my,mz
            if isinstance(a, dict):
                k = set(a.keys())
                if {"mx", "my", "mz"} <= k:
                    xyz = (a["mx"], a["my"], a["mz"])
                    break
            # iterable (mx,my,mz)
            if isinstance(a, (tuple, list)) and len(a) == 3:
                x, y, z = a
                if all(isinstance(v, (int, float)) for v in (x, y, z)):
                    xyz = (x, y, z)
                    break

        if mag_obj is not None:
            self.last_mag_data = mag_obj.mag_tesla
            self.sample_count += 1
            return

        if xyz is not None:
            self.last_mag_data = xyz
            self.sample_count += 1
            return
        # nada útil → ignore
        return

    def on_imu_received(self, *args) -> None:
        """Callback del acelerómetro/IMU (firma flexible).
           Acepta:
             - (sample)
             - (stream_label, sample)
             - (stream_label, ts, sample)
             - (samples_list, imu_idx)  # lista de objetos con .accel_mps2 + índice
             - (stream_label, samples_list, imu_idx)
             - así como variantes donde algún arg sea (ax, ay, az) o {'ax','ay','az'}.
        """
        if not args:
            return

        sample_obj = None
        axayaz = None

        for a in reversed(args):
            # Caso 1: objeto con atributo .accel_mps2
            if hasattr(a, "accel_mps2"):
                sample_obj = a
                break

            # Caso 2: lista/tupla de muestras IMU -> coger la última válida
            if isinstance(a, (list, tuple)) and len(a) > 0:
                last = a[-1]
                if hasattr(last, "accel_mps2"):
                    sample_obj = last
                    break
                # También podría venir como lista de tuplas numéricas (ax,ay,az)
                if isinstance(last, (tuple, list)) and len(last) == 3 and all(isinstance(v, (int, float)) for v in last):
                    axayaz = tuple(last)
                    break

            # Caso 3: dict con ax/ay/az
            if isinstance(a, dict):
                keys = set(a.keys())
                if {"ax", "ay", "az"} <= keys:
                    axayaz = (a["ax"], a["ay"], a["az"])
                    break

            # Caso 4: iterable numérico (ax, ay, az)
            if isinstance(a, (tuple, list)) and len(a) == 3 and all(isinstance(v, (int, float)) for v in a):
                axayaz = tuple(a)
                break

        if sample_obj is not None:
            self.last_accel_data = sample_obj.accel_mps2
            self.imu_count += 1
            if (self.imu_count % 50) == 0:
                try:
                    ax, ay, az = self.last_accel_data
                    print(f"[IMU] muestras: {self.imu_count}  última=({ax:.2f},{ay:.2f},{az:.2f})")
                except Exception:
                    print(f"[IMU] muestras: {self.imu_count}")
            return

        if axayaz is not None:
            self.last_accel_data = axayaz
            self.imu_count += 1
            if (self.imu_count % 50) == 0:
                try:
                    ax, ay, az = self.last_accel_data
                    print(f"[IMU] muestras: {self.imu_count}  última=({ax:.2f},{ay:.2f},{az:.2f})")
                except Exception:
                    print(f"[IMU] muestras: {self.imu_count}")
            return
        # Si no se pudo extraer nada útil, lo ignoramos
        return

    def compute_tilt_comp_heading(self, mx, my, mz, ax, ay, az):
        norm = math.sqrt(ax*ax + ay*ay + az*az) or 1.0
        ax, ay, az = ax/norm, ay/norm, az/norm
        hx = mx*az - mz*ax
        hy = my*az - mz*ay
        heading = math.degrees(math.atan2(hy, hx))
        heading = (heading + self.DECLINATION_DEG + 360.0) % 360.0
        return heading

    def wait_for_new_mag(self, prev_count, timeout=2.0):
        """Espera a que llegue al menos 1 muestra nueva de magnetómetro.
        Devuelve (mx, my, mz) o None si expira el timeout."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.sample_count > prev_count and self.last_mag_data is not None:
                return self.last_mag_data
            time.sleep(0.01)
        return None

    def wait_for_new_imu(self, prev_count, timeout=2.0):
        """Espera un nuevo sample de acelerómetro/IMU y devuelve (ax, ay, az) o None."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.imu_count > prev_count and self.last_accel_data is not None:
                return self.last_accel_data
            time.sleep(0.01)
        return None

    def complete_exhaustive_cardinal_test(self):
        """Test exhaustivo completo de TODAS las combinaciones"""
        print(f"\n🧭 TEST EXHAUSTIVO COMPLETO - TODAS LAS COMBINACIONES")
        print("=" * 80)
        print(f"Probando {len(self.all_combinations)} combinaciones diferentes")
        print("=" * 80)
        
        directions = [("NORTE", 0), ("ESTE", 90), ("SUR", 180), ("OESTE", 270)]
        cardinal_measurements = {}
        
        # Recopilar datos
        for direction, expected in directions:
            input(f"\n👉 Apunta hacia el {direction} y presiona Enter...")

            # Espera muestras nuevas y sincroniza magnetómetro (+ IMU si llega)
            prev_mag = self.sample_count
            prev_imu = self.imu_count
            mxmy_mz = self.wait_for_new_mag(prev_mag, timeout=2.0)
            axyz = self.wait_for_new_imu(prev_imu, timeout=0.8)  # IMU opcional, timeout más corto

            if mxmy_mz is None:
                print(f"❌ Sin datos nuevos de magnetómetro para {direction} (timeout). Gira ligeramente y vuelve a intentar.")
                continue

            mx, my, mz = mxmy_mz

            if axyz is not None:
                ax, ay, az = axyz
                has_imu = True
            else:
                ax = ay = az = None
                has_imu = False

            entry = {'expected': expected, 'mx': mx, 'my': my, 'mz': mz}
            if has_imu:
                entry.update({'ax': ax, 'ay': ay, 'az': az})
            cardinal_measurements[direction] = entry

            mag = math.sqrt(mx*mx + my*my + mz*mz)
            if has_imu:
                g  = math.sqrt(ax*ax + ay*ay + az*az)
                print(f"✅ {direction}: Mag X={mx:.2f}, Y={my:.2f}, Z={mz:.2f} μT (|B|={mag:.1f}) | "
                      f"Acc X={ax:.2f}, Y={ay:.2f}, Z={az:.2f} m/s² (|g|={g:.1f})")
            else:
                print(f"✅ {direction}: Mag X={mx:.2f}, Y={my:.2f}, Z={mz:.2f} μT (|B|={mag:.1f}) | "
                      f"Acc: --- (no hay muestras nuevas)")
        
        if len(cardinal_measurements) < 4:
            print("❌ Faltan datos")
            return
        
        # Análisis exhaustivo
        print(f"\n📊 ANÁLISIS EXHAUSTIVO:")
        print("=" * 90)
        print(f"{'Combinación':>18} | {'Norte':>6} | {'Este':>6} | {'Sur':>6} | {'Oeste':>6} | {'Avg Err':>7} | {'Calidad'}")
        print("-" * 90)
        
        valid_results = []
        
        for i, (combo_name, calc_func) in enumerate(self.all_combinations):
            angles = []
            errors = []
            
            try:
                all_valid = True
                for direction in ['NORTE', 'ESTE', 'SUR', 'OESTE']:
                    data = cardinal_measurements[direction]
                    expected = data['expected']
                    mx, my, mz = data['mx'], data['my'], data['mz']
                    
                    # Calcular ángulo
                    result = calc_func(mx, my, mz)
                    
                    if result is None:
                        all_valid = False
                        break
                    
                    # Usa la IMU capturada en el mismo cardinal si está disponible
                    if all(k in data for k in ('ax','ay','az')):
                        ax, ay, az = data['ax'], data['ay'], data['az']
                        angle_deg = self.compute_tilt_comp_heading(mx, my, mz, ax, ay, az)
                    else:
                        angle_deg = (math.degrees(result) + 360) % 360
                        angle_deg = (angle_deg + self.DECLINATION_DEG + 360) % 360
                    
                    # Verificar que el resultado es válido
                    if not (0 <= angle_deg <= 360) or math.isnan(angle_deg) or math.isinf(angle_deg):
                        all_valid = False
                        break
                    
                    # Calcular error
                    error = min(
                        abs(angle_deg - expected),
                        abs(angle_deg - expected + 360),
                        abs(angle_deg - expected - 360)
                    )
                    
                    angles.append(angle_deg)
                    errors.append(error)
                
                if not all_valid or len(angles) != 4:
                    print(f"{combo_name:>18} | {'ERR':>5} | {'ERR':>5} | {'ERR':>5} | {'ERR':>5} | {'ERR':>6} | ❌ INVÁLIDA")
                    continue
                
                avg_error = sum(errors) / len(errors)
                
                # Clasificar calidad
                if avg_error < 5:
                    quality = "🌟 PERFECTA"
                elif avg_error < 15:
                    quality = "✅ EXCELENTE"
                elif avg_error < 30:
                    quality = "⚠️ BUENA"
                elif avg_error < 60:
                    quality = "🔶 REGULAR"
                else:
                    quality = "❌ MALA"
                
                print(f"{combo_name:>18} | {angles[0]:5.0f}° | {angles[1]:5.0f}° | {angles[2]:5.0f}° | {angles[3]:5.0f}° | {avg_error:6.1f}° | {quality}")
                valid_results.append((combo_name, avg_error, angles, errors))
                
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                print(f"{combo_name:>18} | {'ERR':>5} | {'ERR':>5} | {'ERR':>5} | {'ERR':>5} | {'ERR':>6} | ❌ ERROR")
        
        print("-" * 90)
        
        # Análisis de resultados
        if valid_results:
            valid_results.sort(key=lambda x: x[1])
            
            print(f"\n🏆 TOP 10 MEJORES COMBINACIONES:")
            print("=" * 70)
            
            for i, (combo_name, avg_error, angles, errors) in enumerate(valid_results[:10]):
                print(f"\n{i+1:2d}. {combo_name} (Error promedio: {avg_error:.1f}°)")
                
                directions_list = ['NORTE', 'ESTE', 'SUR', 'OESTE']
                for j, direction in enumerate(directions_list):
                    expected = cardinal_measurements[direction]['expected']
                    actual = angles[j]
                    error = errors[j]
                    status = "✅" if error < 15 else "⚠️" if error < 30 else "❌"
                    print(f"    {direction:>6}: {actual:6.1f}° (error: {error:5.1f}°) {status}")
                
                if i == 0:  # La mejor
                    print(f"    🎯 ¡ESTA ES LA MEJOR COMBINACIÓN!")
                    
                    # Generar código
                    print(f"\n💻 CÓDIGO PYTHON:")
                    print("```python")
                    print("def calculate_heading(mx, my, mz):")
                    print(f"    # Combinación: {combo_name}")
                    
                    # Aquí necesitarías implementar cada combinación específica
                    # Por simplicidad, muestro la estructura general
                    print(f"    # Implementar la fórmula específica aquí")
                    print(f"    heading_deg = (math.degrees(result) + 360) % 360")
                    print(f"    return heading_deg")
                    print("```")
        
        else:
            print(f"\n❌ NINGUNA COMBINACIÓN VÁLIDA ENCONTRADA")
            print("El magnetómetro tiene problemas fundamentales")
        
        return valid_results

def main():
    print("🔍 COMPLETE EXHAUSTIVE MAGNETOMETER TEST")
    print("=" * 60)
    
    try:
        test = CompleteExhaustiveMagnetometerTest()
        
        # Setup device
        device_manager = DeviceManager()
        device_manager.connect()
        device_manager.start_streaming()
        
        class SimpleObserver:
            def __init__(self, test_instance):
                self.test = test_instance

            def on_magneto_received(self, *args):
                try:
                    self.test.on_magneto_received(*args)
                except Exception:
                    pass

            def on_imu_received(self, *args):
                try:
                    self.test.on_imu_received(*args)
                except Exception:
                    pass

            def on_image_received(self, *args): pass
            def on_baro_received(self, *args): pass
            def on_streaming_client_failure(self, *args): pass
        
        observer = SimpleObserver(test)
        device_manager.register_observer(observer)
        device_manager.subscribe()
        
        print("✅ Conectado, esperando datos...")
        time.sleep(2)
        
        # Ir directo al test exhaustivo
        print("Iniciando test exhaustivo automáticamente...")
        test.complete_exhaustive_cardinal_test()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'device_manager' in locals():
            device_manager.cleanup()
        print("✅ Test terminado")

if __name__ == "__main__":
    main()