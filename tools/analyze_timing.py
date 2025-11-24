#!/usr/bin/env python3
"""
Analiza los timing logs detallados del main loop
"""
import json
import sys
from pathlib import Path

def analyze_timing(session_path):
    """Analiza el timing breakdown de una sesión"""
    perf_file = Path(session_path) / "performance.jsonl"
    
    if not perf_file.exists():
        print(f"❌ No existe: {perf_file}")
        return
    
    # Leer todos los frames con timing
    frames_with_timing = []
    total_frames = 0
    
    with open(perf_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            total_frames += 1
            if 'timing' in data:
                frames_with_timing.append(data)
    
    if not frames_with_timing:
        print(f"⚠️  No hay timing breakdown en esta sesión")
        print(f"   Total frames: {total_frames}")
        return
    
    print(f"📊 Análisis de {len(frames_with_timing)} frames con timing detallado")
    print(f"   (de {total_frames} frames totales)\n")
    
    # Calcular promedios y detectar spikes
    timing_sums = {}
    all_totals = []
    spike_frames = []
    
    for frame in frames_with_timing:
        timing = frame['timing']
        all_totals.append(timing['total'])
        for key, value in timing.items():
            if key not in timing_sums:
                timing_sums[key] = []
            timing_sums[key].append(value)
    
    # Detectar spikes (frames >2x promedio)
    avg_total = sum(all_totals) / len(all_totals)
    spike_threshold = avg_total * 2.0
    
    for frame in frames_with_timing:
        if frame['timing']['total'] > spike_threshold:
            spike_frames.append(frame)
    
    # Mostrar resultados
    print("=" * 70)
    print("TIMING BREAKDOWN - Promedios (ms)")
    print("=" * 70)
    
    components = [
        ('get_rgb', 'Observer: get RGB frame'),
        ('get_slam', 'Observer: get SLAM frames'),
        ('get_motion', 'Observer: get motion data'),
        ('build_dict', 'Build frames_dict'),
        ('process_frame', '🔥 Pipeline.process (GPU workers)'),
        ('slam_handling', 'Handle SLAM frames'),
        ('get_results', 'Get depth/events'),
        ('render_slam', 'Render SLAM overlays'),
        ('update_display', '🔥 PresentationManager.update_display'),
        ('total', '═══ TOTAL ITERATION ═══'),
    ]
    
    observer_total = 0
    render_total = 0
    
    for key, label in components:
        if key in timing_sums:
            values = timing_sums[key]
            avg = sum(values) / len(values)
            min_val = min(values)
            max_val = max(values)
            
            if key in ['get_rgb', 'get_slam', 'get_motion']:
                observer_total += avg
            elif key in ['render_slam', 'update_display']:
                render_total += avg
            
            marker = "🔥" if key in ['process_frame', 'update_display'] else "  "
            if key == 'total':
                print("-" * 70)
                print(f"{marker} {label:40s}: {avg:6.2f}ms  (min={min_val:.1f} max={max_val:.1f})")
                theoretical_fps = 1000 / avg
                print(f"   → Theoretical max FPS: {theoretical_fps:.1f}")
            else:
                pct = (avg / timing_sums['total'][0] * 100) if 'total' in timing_sums else 0
                print(f"{marker} {label:40s}: {avg:6.2f}ms  ({pct:4.1f}%)")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    process_frame_avg = sum(timing_sums['process_frame']) / len(timing_sums['process_frame'])
    update_display_avg = sum(timing_sums['update_display']) / len(timing_sums['update_display'])
    total_avg = sum(timing_sums['total']) / len(timing_sums['total'])
    
    print(f"Observer operations:    {observer_total:6.2f}ms  ({observer_total/total_avg*100:4.1f}%)")
    print(f"Pipeline.process:       {process_frame_avg:6.2f}ms  ({process_frame_avg/total_avg*100:4.1f}%) ← GPU workers")
    print(f"Rendering/Display:      {render_total:6.2f}ms  ({render_total/total_avg*100:4.1f}%)")
    print(f"Other:                  {total_avg - observer_total - process_frame_avg - render_total:6.2f}ms")
    print(f"\nTotal iteration:        {total_avg:6.2f}ms")
    print(f"Theoretical FPS:        {1000/total_avg:6.1f} FPS")
    
    # FPS real de la sesión
    actual_fps = frames_with_timing[-1]['fps']
    print(f"Actual FPS:             {actual_fps:6.1f} FPS")
    efficiency = (actual_fps / (1000/total_avg)) * 100
    print(f"Efficiency:             {efficiency:6.1f}%")
    
    print("\n" + "=" * 70)
    print("BOTTLENECK IDENTIFICATION")
    print("=" * 70)
    
    bottlenecks = [
        ('process_frame', process_frame_avg, 'GPU workers (depth + YOLO)'),
        ('update_display', update_display_avg, 'Rendering + OpenCV display'),
        ('observer', observer_total, 'Getting frames from Aria SDK'),
    ]
    
    bottlenecks.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, time_ms, desc) in enumerate(bottlenecks, 1):
        pct = (time_ms / total_avg) * 100
        print(f"{i}. {desc:40s}: {time_ms:6.2f}ms ({pct:4.1f}%)")
    
    print("\n💡 Recommendations:")
    if process_frame_avg > 40:
        print("  • Pipeline.process taking >40ms - GPU workers may be overloaded")
    if update_display_avg > 20:
        print("  • Display update >20ms - consider reducing rendering complexity")
    if observer_total > 5:
        print("  • Observer operations >5ms - frame acquisition overhead")
    
    # Analyze spikes
    if spike_frames:
        print(f"\n⚠️  SPIKE ANALYSIS - {len(spike_frames)} frames exceeded 2x average")
        print("=" * 70)
        print(f"Spike threshold: {spike_threshold:.1f}ms (2x average of {avg_total:.1f}ms)")
        
        # Analyze what caused spikes
        spike_causes = {}
        for spike in spike_frames:
            timing = spike['timing']
            # Find the component that took longest in this spike
            components = [(k, v) for k, v in timing.items() if k != 'total']
            slowest = max(components, key=lambda x: x[1])
            cause = slowest[0]
            if cause not in spike_causes:
                spike_causes[cause] = []
            spike_causes[cause].append(slowest[1])
        
        print("\nSpike causes (most common component):")
        for cause, times in sorted(spike_causes.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(times)
            avg_time = sum(times) / len(times)
            print(f"  {cause:20s}: {count:3d} spikes (avg {avg_time:.1f}ms)")
        
        # Show worst 3 spikes
        spike_frames.sort(key=lambda x: x['timing']['total'], reverse=True)
        print(f"\nWorst 3 spikes:")
        for i, spike in enumerate(spike_frames[:3], 1):
            frame_num = spike['frame_number']
            total = spike['timing']['total']
            timing = spike['timing']
            components = [(k, v) for k, v in timing.items() if k != 'total']
            components.sort(key=lambda x: x[1], reverse=True)
            print(f"  {i}. Frame {frame_num}: {total:.1f}ms")
            for name, time_val in components[:3]:
                print(f"       {name:20s}: {time_val:6.1f}ms")
    
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_timing.py <session_path>")
        print("\nExample:")
        print("  python analyze_timing.py logs/session_1763989277816")
        sys.exit(1)
    
    analyze_timing(sys.argv[1])
