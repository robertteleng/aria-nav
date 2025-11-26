#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌐 Web Dashboard - Presentation Layer
Dashboard web para integration en presentation/dashboards/

Siguiendo la arquitectura: presentation/dashboards/web_dashboard.py
Junto con: opencv_dashboard.py y rerun_dashboard.py
"""

from flask import Flask, Response, jsonify
import cv2
import numpy as np
import time
import threading
from typing import Optional, Dict, List


class WebDashboard:
    """
    Web Dashboard para presentation layer - Compatible con Observer pattern
    """
    
    def __init__(self, host='0.0.0.0', port=8080):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        
        # Frame storage (thread-safe)
        self._frame_lock = threading.Lock()
        self.current_rgb_frame = None
        self.current_depth_frame = None
        self.current_slam1_frame = None  
        self.current_slam2_frame = None
        
        # Pending detections for current frame (cleared when new frame arrives)
        self._pending_detections = []
        
        # Placeholder cache (avoid recreating constantly)
        self._placeholder_cache = {}
        
        # Performance stats
        self.stats = {
            'fps': 0.0,
            'detections_total': 0,
            'audio_commands': 0,
            'uptime': 0.0,
            'frames_processed': 0,
            'slam1_events': 0,
            'slam2_events': 0,
            'current_detections_count': 0,
            'critical_beeps': 0,
            'normal_beeps': 0,
            'critical_frequency': 0,
            'normal_frequency': 0,
        }
        self.slam_messages: List[str] = []
        # System logs with levels
        self.logs = []
        self.max_logs = 100
        
        # Recent detections
        self.recent_detections = []
        self.max_recent_detections = 10
        
        self.start_time = time.time()
        self.setup_routes()
        
        print(f"🌐 WebDashboard initialized on {host}:{port}")
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return self._get_dashboard_html()
        
        @self.app.route('/stats')
        def get_stats():
            try:
                self.stats['uptime'] = time.time() - self.start_time
                payload = dict(self.stats)
                payload['slam_messages'] = self.slam_messages[-6:]
                # Debug: print stats to console occasionally
                if payload.get('frames_processed', 0) % 100 == 0 and payload.get('frames_processed', 0) > 0:
                    print(f"[WEB] Stats endpoint: FPS={payload['fps']:.1f}, Frames={payload['frames_processed']}, Det={payload['detections_total']}")
                return jsonify(payload)
            except Exception as e:
                print(f"[WEB ERROR] Stats endpoint failed: {e}")
                return jsonify({'error': str(e), 'fps': 0, 'frames_processed': 0}), 500
        
        @self.app.route('/logs')  
        def get_logs():
            return jsonify({'logs': self.logs[-30:]})
        
        @self.app.route('/detections')
        def get_detections():
            try:
                # Convert DetectedObject dataclasses to JSON-serializable dicts
                serializable_detections = []
                for det in self.recent_detections:
                    if isinstance(det, dict):
                        # Already a dict, ensure numpy types are converted
                        det_dict = {k: (float(v) if hasattr(v, 'item') else v) 
                                   for k, v in det.items()}
                    else:
                        # DetectedObject dataclass - convert to dict with native Python types
                        det_dict = {
                            'name': det.name,
                            'confidence': float(det.confidence),
                            'bbox': [int(x) for x in det.bbox],
                            'center_x': float(det.center_x),
                            'center_y': float(det.center_y),
                            'area': float(det.area),
                            'zone': det.zone,
                            'distance_bucket': det.distance_bucket,
                            'relevance_score': float(det.relevance_score),
                            'depth_value': float(det.depth_value) if det.depth_value is not None else None,
                            'depth_raw': float(det.depth_raw) if det.depth_raw is not None else None,
                        }
                    serializable_detections.append(det_dict)
                
                return jsonify({
                    'recent_detections': serializable_detections,
                    'detection_count': len(serializable_detections)
                })
            except Exception as e:
                print(f"[WEB ERROR] /detections endpoint failed: {e}")
                return jsonify({'recent_detections': [], 'detection_count': 0, 'error': str(e)}), 500
        
        @self.app.route('/video_feed/<stream_type>')
        def video_feed(stream_type):
            def generate_frames():
                while True:
                    frame = None
                    pending_dets = []
                    
                    with self._frame_lock:
                        if stream_type == 'rgb':
                            frame = self.current_rgb_frame
                            pending_dets = self._pending_detections.copy()
                        elif stream_type == 'depth':
                            frame = self.current_depth_frame
                        elif stream_type == 'slam1':
                            frame = self.current_slam1_frame
                        elif stream_type == 'slam2':
                            frame = self.current_slam2_frame
                    
                    if frame is not None:
                        # NOTE: El frame ya viene con bounding boxes dibujados por frame_renderer
                        # No necesitamos dibujarlos de nuevo aquí
                        
                        # Resize for web efficiency
                        height, width = frame.shape[:2]
                        if width > 640:
                            scale = 640 / width
                            new_width = int(width * scale)
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Encode with good compression
                        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        if success:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        # Send placeholder
                        placeholder = self._create_placeholder_frame(stream_type)
                        success, buffer = cv2.imencode('.jpg', placeholder)
                        if success:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    time.sleep(0.1)  # 10 FPS for web
            
            return Response(generate_frames(), 
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def _get_dashboard_html(self):
        """Generate dashboard HTML - Professional design without emojis"""
        try:
            from presentation.dashboards.dashboard_html_template import DASHBOARD_HTML
            return DASHBOARD_HTML
        except ImportError:
            # Fallback to inline HTML if template file is missing
            return self._get_fallback_html()
    
    def _get_fallback_html(self):
        """Fallback HTML if template not available"""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Aria Navigation Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 0; 
            background: linear-gradient(135deg, #1a1a1a, #2d2d2d); 
            color: white; 
            overflow-x: hidden;
        }
        .header { 
            background: linear-gradient(90deg, #007acc, #005a9e); 
            padding: 15px; 
            text-align: center; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .header h1 { margin: 0; font-size: 24px; }
        .controls { 
            display: flex; 
            justify-content: center; 
            gap: 10px; 
            margin: 15px 0; 
            flex-wrap: wrap;
        }
        .grid { 
            display: grid; 
            grid-template-columns: repeat(3, minmax(320px, 1fr)); 
            grid-auto-rows: auto; 
            gap: 15px; 
            padding: 15px; 
            max-width: 1400px; 
            margin: 0 auto;
        }
        .system-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 10px;
        }
        .system-metrics .stat {
            background: linear-gradient(145deg, #232741, #1b1f35);
            border: 1px solid #2f3453;
        }
        .panel { 
            background: linear-gradient(145deg, #2a2a2a, #343434); 
            border-radius: 12px; 
            padding: 20px; 
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border: 1px solid #404040;
        }
        .panel h3 { 
            margin: 0 0 15px 0; 
            color: #007acc; 
            font-size: 18px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .frame { 
            max-width: 100%; 
            height: auto; 
            border-radius: 8px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .frame-half {
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }
        .stats-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); 
            gap: 10px; 
        }
        .stat { 
            background: linear-gradient(145deg, #3a3a3a, #444444); 
            padding: 15px; 
            border-radius: 8px; 
            text-align: center;
            border: 1px solid #505050;
        }
        .stat-value { 
            font-size: 24px; 
            font-weight: bold; 
            color: #00ff88;
            margin-bottom: 5px;
        }
        .stat-label { 
            font-size: 12px; 
            opacity: 0.8; 
            text-transform: uppercase;
        }
        .logs { 
            height: 300px; 
            overflow-y: auto; 
            background: #000; 
            padding: 15px; 
            font-family: 'Consolas', monospace; 
            border-radius: 8px;
            border: 1px solid #333;
            font-size: 13px;
        }
        .log-entry { margin: 2px 0; padding: 2px 0; }
        .log-info { color: #00ff88; }
        .log-warn { color: #ffaa00; }
        .log-error { color: #ff4444; }
        .log-audio { color: #ff88ff; }
        .log-detect { color: #00aaff; }
        .log-system { color: #88ff88; }
        button { 
            padding: 12px 20px; 
            background: linear-gradient(145deg, #007acc, #005a9e); 
            color: white; 
            border: none; 
            border-radius: 6px; 
            cursor: pointer; 
            font-weight: bold;
            transition: all 0.3s ease;
        }
        button:hover { 
            background: linear-gradient(145deg, #0088dd, #006bb0); 
            transform: translateY(-1px);
        }
        .status { 
            padding: 8px 15px; 
            background: #333; 
            border-radius: 20px; 
            font-size: 14px;
            border: 1px solid #555;
        }
        .status.online { background: #006600; }
        .detection-item {
            background: #2a2a2a;
            margin: 5px 0;
            padding: 8px;
            border-radius: 4px;
            border-left: 3px solid #007acc;
            font-size: 14px;
        }
        .no-data {
            text-align: center;
            opacity: 0.6;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Aria Navigation Web Dashboard</h1>
        <div class="controls">
            <button onclick="location.reload()">🔄 Refresh</button>
            <button onclick="toggleAutoRefresh()" id="auto-btn">⏯️ Auto Refresh</button>
            <span class="status" id="status">Status: Loading...</span>
        </div>
    </div>
    
    <div class="grid">
        <!-- Video Feeds -->
        <div class="panel">
            <h3>📹 RGB + YOLO Detection</h3>
            <img id="rgb-frame" class="frame" src="/video_feed/rgb" alt="RGB Feed">
        </div>
        
        <div class="panel">
            <h3>🗺️ Depth Map</h3>
            <img id="depth-frame" class="frame" src="/video_feed/depth" alt="Depth Feed">
        </div>

        <div class="panel">
            <h3>👁️ Visión Periférica (SLAM)</h3>
            <div style="display:flex; gap:10px; flex-wrap:wrap; justify-content:space-between;">
                <div style="flex:1; min-width:200px;">
                    <h4>SLAM Izquierda</h4>
                    <img id="slam1-frame" class="frame-half" src="/video_feed/slam1" alt="SLAM1 Feed">
                </div>
                <div style="flex:1; min-width:200px;">
                    <h4>SLAM Derecha</h4>
                    <img id="slam2-frame" class="frame-half" src="/video_feed/slam2" alt="SLAM2 Feed">
                </div>
            </div>
            <div id="slam-events" style="margin-top:10px; font-size:0.85em; color:#ccc;">
                <div>Eventos laterales aparecerán aquí…</div>
            </div>
        </div>

        <div class="panel">
            <h3>📝 System Logs</h3>
            <div class="logs" id="logs-container">Loading logs...</div>
        </div>

        <div class="panel">
            <h3>📊 Performance Metrics</h3>
            <div class="stats-grid" id="stats-container">
                <div class="stat">
                    <div class="stat-value" id="fps">0</div>
                    <div class="stat-label">FPS</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="detections">0</div>
                    <div class="stat-label">Detecciones</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="audio">0</div>
                    <div class="stat-label">Alertas audio</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="slam1-count">0</div>
                    <div class="stat-label">SLAM1 det.</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="slam2-count">0</div>
                    <div class="stat-label">SLAM2 det.</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="uptime">0s</div>
                    <div class="stat-label">Uptime</div>
                </div>
                <div class="stat">
                    <div class="stat-value" id="frames">0</div>
                    <div class="stat-label">Frames</div>
                </div>
            </div>
        </div>


        <div class="panel">
            <h3>🎯 Detecciones recientes</h3>
            <div id="detections-container">
                <div class="no-data">Aún no hay detecciones…</div>
            </div>
        </div>
    </div>
    
    <script>
        let autoRefresh = false;
        let refreshInterval;
        
        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            const btn = document.getElementById('auto-btn');
            const status = document.getElementById('status');
            
            if (autoRefresh) {
                status.textContent = 'Status: Auto-refresh ON';
                status.className = 'status online';
                btn.textContent = '⏸️ Stop Auto';
                startAutoRefresh();
            } else {
                status.textContent = 'Status: Auto-refresh OFF';
                status.className = 'status';
                btn.textContent = '▶️ Start Auto';
                if (refreshInterval) {
                    clearInterval(refreshInterval);
                }
            }
        }
        
        function startAutoRefresh() {
            if (!autoRefresh) return;
            updateData();
            refreshInterval = setInterval(updateData, 2000);
        }
        
        function updateData() {
            
            // Update stats
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    // Debug log
                    console.log('Stats update:', data);
                    
                    document.getElementById('fps').textContent = data.fps ? data.fps.toFixed(1) : '0.0';
                    document.getElementById('detections').textContent = data.detections_total || 0;
                    document.getElementById('audio').textContent = data.audio_commands || 0;
                    document.getElementById('slam1-count').textContent = data.slam1_events || 0;
                    document.getElementById('slam2-count').textContent = data.slam2_events || 0;
                    document.getElementById('frames').textContent = data.frames_processed || 0;
                    
                    const eventsBox = document.getElementById('slam-events');
                    if (data.slam_messages && data.slam_messages.length) {
                        eventsBox.innerHTML = data.slam_messages.map(msg => `<div>${msg}</div>`).join('');
                    } else {
                        eventsBox.innerHTML = '<div>No hay eventos SLAM recientes…</div>';
                    }
                    document.getElementById('uptime').textContent = formatUptime(data.uptime || 0);
                })
                .catch(err => {
                    console.error('Stats error:', err);
                    // Show error in UI
                    document.getElementById('fps').textContent = 'ERR';
                });
            
            // Update logs
            fetch('/logs')
                .then(response => response.json())
                .then(data => {
                    const logsContainer = document.getElementById('logs-container');
                    logsContainer.innerHTML = data.logs.map(log => {
                        const level = log.includes('[ERROR]') ? 'log-error' :
                                     log.includes('[WARN]') ? 'log-warn' :
                                     log.includes('[AUDIO]') ? 'log-audio' :
                                     log.includes('[DETECT]') ? 'log-detect' :
                                     log.includes('[SYSTEM]') ? 'log-system' : 'log-info';
                        return `<div class="log-entry ${level}">${log}</div>`;
                    }).join('');
                    logsContainer.scrollTop = logsContainer.scrollHeight;
                })
                .catch(err => console.log('Logs error:', err));
            
            // Update detections
            fetch('/detections')
                .then(response => response.json())
                .then(data => {
                    const detectionsContainer = document.getElementById('detections-container');
                    if (data.recent_detections.length > 0) {
                        detectionsContainer.innerHTML = data.recent_detections.map(det => 
                            `<div class="detection-item">
                                <strong>${det.name}</strong> (${det.confidence.toFixed(2)}) 
                                - ${det.zone} - ${det.distance || 'unknown distance'}
                            </div>`
                        ).join('');
                    } else {
                        detectionsContainer.innerHTML = '<div class="no-data">No recent detections</div>';
                    }
                })
                .catch(err => console.log('Detections error:', err));
        }
        
        function formatUptime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            
            if (hours > 0) {
                return `${hours}h ${minutes}m`;
            } else if (minutes > 0) {
                return `${minutes}m ${secs}s`;
            } else {
                return `${secs}s`;
            }
        }
        
        // Start auto-refresh by default and populate metrics immediately
        window.addEventListener('load', () => {
            updateData();
            setTimeout(() => {
                toggleAutoRefresh();
            }, 1000);
        });
    </script>
</body>
</html>
        '''
    
    def _create_placeholder_frame(self, stream_type: str) -> np.ndarray:
        """Create placeholder frame when no data available (cached)"""
        # Check cache first
        if stream_type in self._placeholder_cache:
            return self._placeholder_cache[stream_type]
        
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(240):
            intensity = int(30 + (y / 240) * 20)
            frame[y, :] = [intensity, intensity, intensity]
        
        # Add text
        text = f'No {stream_type.upper()} Data'
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
        text_x = (320 - text_size[0]) // 2
        text_y = (240 + text_size[1]) // 2
        
        cv2.putText(frame, text, (text_x, text_y), font, 0.7, (100, 100, 100), 2)
        cv2.putText(frame, "Waiting for stream...", (text_x - 20, text_y + 30), 
                   font, 0.5, (80, 80, 80), 1)
        
        # Cache for future use
        self._placeholder_cache[stream_type] = frame
        
        return frame


    
    # =================================================================
    # OBSERVER PATTERN COMPATIBILITY - Same API as OpenCV/Rerun dashboards
    # =================================================================
    
    def log_rgb_frame(self, frame: np.ndarray):
        """Update RGB frame - Observer pattern compatibility
        NOTE: El frame ya viene con bounding boxes dibujados por frame_renderer"""
        if frame is not None:
            with self._frame_lock:
                # Store frame (already has detections drawn)
                self.current_rgb_frame = frame.copy()
    
    def log_depth_map(self, depth_map: np.ndarray):
        """Update depth frame - Observer pattern compatibility"""  
        if depth_map is not None:
            # Convert depth to colormap for visualization
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            
            with self._frame_lock:
                self.current_depth_frame = depth_colored
    
    def log_slam1_frame(self, slam1_frame: np.ndarray, events: Optional[List[Dict]] = None):
        """Update SLAM1 frame - frame ya viene con detecciones dibujadas"""
        if slam1_frame is not None:
            with self._frame_lock:
                frame_copy = slam1_frame
                if frame_copy.ndim == 2 or (frame_copy.ndim == 3 and frame_copy.shape[2] == 1):
                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
                # NO dibujar events, ya vienen dibujados del renderer
                if events:
                    self.stats['slam1_events'] = len(events)
                    for event in events:
                        desc = f"SLAM1: {event.get('name', 'obj')} {event.get('distance', '')} ({event.get('zone', '')})"
                        self._append_slam_message(desc)
                else:
                    self.stats['slam1_events'] = 0
                self.current_slam1_frame = frame_copy

    def log_slam2_frame(self, slam2_frame: np.ndarray, events: Optional[List[Dict]] = None):
        """Update SLAM2 frame - frame ya viene con detecciones dibujadas"""
        if slam2_frame is not None:
            with self._frame_lock:
                frame_copy = slam2_frame
                if frame_copy.ndim == 2 or (frame_copy.ndim == 3 and frame_copy.shape[2] == 1):
                    frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)
                # NO dibujar events, ya vienen dibujados del renderer
                if events:
                    self.stats['slam2_events'] = len(events)
                    for event in events:
                        desc = f"SLAM2: {event.get('name', 'obj')} {event.get('distance', '')} ({event.get('zone', '')})"
                        self._append_slam_message(desc)
                else:
                    self.stats['slam2_events'] = 0
                self.current_slam2_frame = frame_copy
    
    def log_detections(self, detections: List[Dict], frame_shape: Optional[tuple] = None):
        """Log detections - Observer pattern compatibility"""
        if detections:
            # Add timestamp to detections for stats
            for detection in detections:
                if isinstance(detection, dict):
                    detection['timestamp'] = time.strftime("%H:%M:%S")
            
            # Keep recent detections for stats/display
            self.recent_detections.extend(detections)
            if len(self.recent_detections) > self.max_recent_detections:
                self.recent_detections = self.recent_detections[-self.max_recent_detections:]
            
            # NOTE: NO dibujamos aquí porque el frame_renderer ya dibujó los bounding boxes
            # El frame que llega ya tiene las detecciones anotadas
            
            # Update stats
            self.stats['detections_total'] += len(detections)
            self.stats['current_detections_count'] = len(detections)
        else:
            # No detections in this frame
            self.stats['current_detections_count'] = 0
    
    def log_system_message(self, message: str, level: str = "INFO"):
        """Log system message - Observer pattern compatibility"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        
        # Keep only recent logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        # Also print to console
        print(log_entry)
    
    def log_audio_command(self, command: str, priority: int = 5):
        """Log audio command - Observer pattern compatibility"""
        self.stats['audio_commands'] += 1
        self.log_system_message(f"Audio [P{priority}]: {command}", "AUDIO")
    
    def log_motion_state(self, motion_state: str, imu_magnitude: float):
        """Log motion state - Observer pattern compatibility"""
        self.log_system_message(f"Motion: {motion_state.upper()} (mag: {imu_magnitude:.2f})", "SYSTEM")

    def update_all(self):
        """Update method - Observer pattern compatibility (no-op for web)"""
        # Web dashboard updates via AJAX, no need for manual update
        return 255  # Return dummy key value

    def shutdown(self):
        """Shutdown web dashboard - Observer pattern compatibility"""
        duration = (time.time() - self.start_time) / 60.0
        self.log_system_message(f"System shutting down - Session: {duration:.1f}min", "SYSTEM")
        print(f"🌐 Web Dashboard session: {duration:.1f}min")

    def _draw_slam_events(self, frame: np.ndarray, events: List[Dict], color: tuple) -> None:
        if frame is None or not events:
            return
        height, width = frame.shape[:2]
        for event in events:
            bbox = event.get('bbox')
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = event.get('name', 'obj')
            distance = event.get('distance')
            if distance and distance not in {'', 'unknown'}:
                label = f"{label} {distance}"
            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_rgb_detections(self, frame: np.ndarray, detections: List) -> None:
        """Draw RGB detections on frame with bounding boxes and labels"""
        if frame is None or not detections:
            return
        
        height, width = frame.shape[:2]
        
        for det in detections:
            # Handle both dict and DetectedObject dataclass
            if isinstance(det, dict):
                # Skip SLAM detections (only draw RGB)
                if det.get('camera_source') in ['slam1', 'slam2']:
                    continue
                bbox = det.get('bbox')
                name = det.get('name', det.get('class_name', 'object'))
                confidence = det.get('confidence')
                distance_str = det.get('distance_bucket', det.get('distance', ''))
            else:
                # DetectedObject dataclass
                bbox = getattr(det, 'bbox', None)
                name = getattr(det, 'name', 'object')
                confidence = getattr(det, 'confidence', None)
                distance_str = getattr(det, 'distance_bucket', '')
            
            if not bbox or len(bbox) != 4:
                continue
            
            # Get bbox coordinates
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Clamp to frame boundaries
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width - 1, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height - 1, y2))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Choose color based on distance
            if distance_str in ['very_close']:
                color = (0, 0, 255)  # Red for very close
            elif distance_str in ['close']:
                color = (0, 165, 255)  # Orange for close
            elif distance_str in ['medium']:
                color = (0, 255, 255)  # Yellow for medium
            else:
                color = (0, 255, 0)  # Green for far
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = name
            if confidence is not None:
                label = f"{label} {confidence:.0%}"
            
            if distance_str and distance_str not in {'', 'unknown'}:
                label = f"{label} {distance_str}"
            
            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 4),
                (x1 + text_width, y1),
                color,
                -1  # Filled
            )
            
            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White text
                1,
                cv2.LINE_AA,
            )

    def _append_slam_message(self, message: str) -> None:
        if not message:
            return
        self.slam_messages.append(f"[{time.strftime('%H:%M:%S')}] {message}")
        if len(self.slam_messages) > 20:
            self.slam_messages = self.slam_messages[-20:]
    
    # =================================================================
    # WEB DASHBOARD SPECIFIC METHODS
    # =================================================================
    
    def start_server(self):
        """Start Flask server in background thread with proper startup verification"""
        import urllib.request
        
        def run_server():
            self.log_system_message(f"Web dashboard starting on {self.host}:{self.port}", "SYSTEM")
            try:
                # Disable Flask startup messages
                import logging
                log = logging.getLogger('werkzeug')
                log.setLevel(logging.ERROR)
                
                self.app.run(host=self.host, port=self.port, debug=False, threaded=True, use_reloader=False)
            except Exception as e:
                self.log_system_message(f"Web server error: {e}", "ERROR")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Esperar a que el servidor esté realmente listo con HTTP check
        url = f"http://{self.host}:{self.port}/stats"
        max_attempts = 30  # 15 segundos máximo
        
        print(f"🔄 Esperando a que el servidor web esté listo...")
        for attempt in range(max_attempts):
            try:
                response = urllib.request.urlopen(url, timeout=1)
                if response.getcode() == 200:
                    # Servidor está respondiendo correctamente
                    self.log_system_message("🌐 Web dashboard ready", "SYSTEM")
                    self.log_system_message(f"🔗 Local access: http://localhost:{self.port}", "SYSTEM")
                    print(f"✅ Servidor web listo en http://localhost:{self.port}")
                    return server_thread
            except Exception:
                pass
            time.sleep(0.5)
        
        # Si llegamos aquí, el servidor tardó mucho
        print("⚠️  Web dashboard tardando en iniciar, pero debería estar listo pronto...")
        return server_thread
    
    def update_performance_stats(self, fps: float = 0.0, frames_processed: int = 0, coordinator_stats: Optional[Dict] = None):
        """Update performance statistics including beep stats from coordinator"""
        self.stats['fps'] = fps
        self.stats['frames_processed'] = frames_processed
        
        # Update beep stats if available in coordinator_stats
        if coordinator_stats:
            self.stats['critical_beeps'] = coordinator_stats.get('critical_beeps', 0)
            self.stats['normal_beeps'] = coordinator_stats.get('normal_beeps', 0)
            self.stats['critical_frequency'] = coordinator_stats.get('critical_frequency', 0)
            self.stats['normal_frequency'] = coordinator_stats.get('normal_frequency', 0)
        
        # Debug output occasionally
        if frames_processed % 100 == 0 and frames_processed > 0:
            print(f"[WEB] Stats: FPS={fps:.1f}, Frames={frames_processed}, Det={self.stats['detections_total']}, " +
                  f"CritBeeps={self.stats['critical_beeps']}, NormBeeps={self.stats['normal_beeps']}")
        # Debug output occasionally
        if frames_processed % 100 == 0 and frames_processed > 0:
            print(f"[WEB] Stats update: FPS={fps:.1f}, Frames={frames_processed}, Det={self.stats['detections_total']}")


# Test function compatible with Observer pattern
def test_web_dashboard():
    """Test web dashboard with simulated data"""
    print("🧪 Testing Web Dashboard...")
    
    dashboard = WebDashboard(port=8080)
    dashboard.start_server()
    
    print("📱 Open browser: http://localhost:8080")
    print("⌨️  Press Ctrl+C to stop")
    
    try:
        # Simulate Observer data flow
        for i in range(200):
            # Simulate RGB frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.putText(test_frame, f"Frame {i}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.rectangle(test_frame, (100, 100), (200, 200), (0, 255, 0), 2)
            
            dashboard.log_rgb_frame(test_frame)
            
            # Simulate detections
            if i % 10 == 0:
                fake_detections = [
                    {
                        'name': 'person',
                        'confidence': 0.85,
                        'zone': 'center',
                        'distance': 'close',
                        'bbox': [100, 100, 200, 200]
                    }
                ]
                dashboard.log_detections(fake_detections)
            
            # Simulate depth map
            if i % 15 == 0:
                fake_depth = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
                dashboard.log_depth_map(fake_depth)
            
            # Simulate system messages
            if i % 20 == 0:
                dashboard.log_system_message(f"Processing frame {i}", "INFO")
                
            if i % 30 == 0:
                dashboard.log_audio_command(f"Test command {i//30}")
            
            # Update performance stats
            dashboard.update_performance_stats(fps=30.0, frames_processed=i)
            
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\n🔄 Shutting down test...")
        dashboard.shutdown()
        print("✅ Web dashboard test completed")


if __name__ == "__main__":
    test_web_dashboard()
