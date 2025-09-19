#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web Dashboard - SSH Compatible
Dashboard web que puedes ver desde Mac via http://jetson_ip:8080
"""

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import time
import threading
import base64
import json
from typing import Optional, Dict, Any

class WebDashboard:
    """
    Dashboard web para ver desde Mac via SSH tunnel
    """
    
    def __init__(self, host='0.0.0.0', port=8080):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        
        # Frame storage
        self.current_rgb_frame = None
        self.current_depth_frame = None
        self.current_slam1_frame = None  
        self.current_slam2_frame = None
        
        # Stats
        self.stats = {
            'fps': 0.0,
            'detections': 0,
            'audio_commands': 0,
            'uptime': 0.0
        }
        
        # Logs
        self.logs = []
        self.max_logs = 50
        
        self.setup_routes()
        self.start_time = time.time()
        
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return '''
<!DOCTYPE html>
<html>
<head>
    <title>Jetson Dashboard</title>
    <style>
        body { font-family: Arial; margin: 0; background: #1a1a1a; color: white; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 10px; }
        .panel { background: #2a2a2a; border-radius: 8px; padding: 15px; }
        .frame { max-width: 100%; height: auto; }
        .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .stat { background: #3a3a3a; padding: 10px; border-radius: 4px; text-align: center; }
        .logs { height: 300px; overflow-y: auto; background: #000; padding: 10px; font-family: monospace; }
        .refresh { margin: 10px 0; }
        button { padding: 10px 20px; background: #007acc; color: white; border: none; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <h1 style="text-align: center;">🚀 Jetson Navigation Dashboard</h1>
    
    <div class="refresh">
        <button onclick="location.reload()">🔄 Refresh</button>
        <button onclick="toggleAutoRefresh()">⏯️ Auto Refresh</button>
        <span id="status">Status: Loading...</span>
    </div>
    
    <div class="grid">
        <div class="panel">
            <h3>📹 RGB + YOLO</h3>
            <img id="rgb-frame" class="frame" src="/video_feed/rgb" alt="RGB Feed">
        </div>
        
        <div class="panel">
            <h3>🗺️ Depth Map</h3>
            <img id="depth-frame" class="frame" src="/video_feed/depth" alt="Depth Feed">
        </div>
        
        <div class="panel">
            <h3>📊 Stats</h3>
            <div class="stats" id="stats-container">
                <div class="stat">FPS: <span id="fps">0</span></div>
                <div class="stat">Detections: <span id="detections">0</span></div>
                <div class="stat">Audio: <span id="audio">0</span></div>
                <div class="stat">Uptime: <span id="uptime">0s</span></div>
            </div>
        </div>
        
        <div class="panel">
            <h3>📝 System Logs</h3>
            <div class="logs" id="logs-container">Loading logs...</div>
        </div>
    </div>
    
    <script>
        let autoRefresh = false;
        
        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            if (autoRefresh) {
                document.getElementById('status').textContent = 'Status: Auto-refresh ON';
                startAutoRefresh();
            } else {
                document.getElementById('status').textContent = 'Status: Auto-refresh OFF';
            }
        }
        
        function startAutoRefresh() {
            if (!autoRefresh) return;
            
            // Update stats
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('fps').textContent = data.fps.toFixed(1);
                    document.getElementById('detections').textContent = data.detections;
                    document.getElementById('audio').textContent = data.audio_commands;
                    document.getElementById('uptime').textContent = data.uptime.toFixed(0) + 's';
                });
            
            // Update logs
            fetch('/logs')
                .then(response => response.json())
                .then(data => {
                    const logsContainer = document.getElementById('logs-container');
                    logsContainer.innerHTML = data.logs.map(log => 
                        '<div>' + log + '</div>'
                    ).join('');
                    logsContainer.scrollTop = logsContainer.scrollHeight;
                });
            
            setTimeout(startAutoRefresh, 1000);
        }
        
        // Start auto-refresh by default
        setTimeout(() => {
            toggleAutoRefresh();
        }, 1000);
    </script>
</body>
</html>
            '''
        
        @self.app.route('/stats')
        def get_stats():
            self.stats['uptime'] = time.time() - self.start_time
            return jsonify(self.stats)
        
        @self.app.route('/logs')  
        def get_logs():
            return jsonify({'logs': self.logs[-20:]})  # Last 20 logs
        
        @self.app.route('/video_feed/<stream_type>')
        def video_feed(stream_type):
            def generate_frames():
                while True:
                    frame = None
                    
                    if stream_type == 'rgb':
                        frame = self.current_rgb_frame
                    elif stream_type == 'depth':
                        frame = self.current_depth_frame
                    elif stream_type == 'slam1':
                        frame = self.current_slam1_frame
                    elif stream_type == 'slam2':
                        frame = self.current_slam2_frame
                    
                    if frame is not None:
                        # Encode frame as JPEG
                        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if success:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    else:
                        # Send black frame if no data
                        black_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                        cv2.putText(black_frame, f'No {stream_type.upper()} data', (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        success, buffer = cv2.imencode('.jpg', black_frame)
                        if success:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    time.sleep(0.1)  # 10 FPS for web
            
            return Response(generate_frames(), 
                          mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def update_rgb_frame(self, frame: np.ndarray):
        """Update RGB frame"""
        if frame is not None:
            self.current_rgb_frame = frame.copy()
    
    def update_depth_frame(self, depth_map: np.ndarray):
        """Update depth frame"""  
        if depth_map is not None:
            # Convert depth to colormap
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            self.current_depth_frame = depth_colored
    
    def update_slam_frame(self, frame: np.ndarray, camera: str):
        """Update SLAM frame"""
        if frame is not None:
            if camera == 'slam1':
                self.current_slam1_frame = frame.copy()
            elif camera == 'slam2':
                self.current_slam2_frame = frame.copy()
    
    def update_stats(self, fps: float, detections: int, audio_commands: int):
        """Update performance stats"""
        self.stats.update({
            'fps': fps,
            'detections': detections, 
            'audio_commands': audio_commands
        })
    
    def log_message(self, message: str, level: str = "INFO"):
        """Add log message"""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        self.logs.append(log_entry)
        
        # Keep only recent logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
    
    def start_server(self):
        """Start Flask server in background thread"""
        def run_server():
            self.log_message(f"Web dashboard starting on {self.host}:{self.port}", "SYSTEM")
            try:
                self.app.run(host=self.host, port=self.port, debug=False, threaded=True)
            except Exception as e:
                self.log_message(f"Web server error: {e}", "ERROR")
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        self.log_message("Web dashboard ready - Access via browser", "SYSTEM")
        return server_thread


# Integration con tu sistema Jetson
class JetsonWithWebDashboard:
    """
    Wrapper que añade web dashboard a tu sistema Jetson existente
    """
    
    def __init__(self, port=8080):
        # Web dashboard
        self.dashboard = WebDashboard(port=port)
        self.dashboard.start_server()
        
        # Stats tracking
        self.frame_count = 0
        self.detection_count = 0
        self.audio_command_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        
        time.sleep(1)  # Let server start
        print(f"🌐 Web dashboard: http://localhost:{port}")
        print(f"🔗 SSH tunnel: ssh -L {port}:localhost:{port} jetson_user@jetson_ip")
    
    def process_frame(self, frame: np.ndarray, detections: list = None) -> np.ndarray:
        """Process frame and update dashboard"""
        
        # Update web dashboard
        self.dashboard.update_rgb_frame(frame)
        
        # Update stats
        self.frame_count += 1
        if detections:
            self.detection_count += len(detections)
            
        # Calculate FPS
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
            
            # Update dashboard stats
            self.dashboard.update_stats(self.fps, self.detection_count, self.audio_command_count)
        
        return frame
    
    def log(self, message: str, level: str = "INFO"):
        """Log message to web dashboard"""
        self.dashboard.log_message(message, level)
        print(f"[{level}] {message}")
    
    def audio_command_sent(self, command: str):
        """Track audio command"""
        self.audio_command_count += 1
        self.log(f"Audio: {command}", "AUDIO")


# Usage example
if __name__ == "__main__":
    # Test web dashboard
    dashboard_system = JetsonWithWebDashboard(port=8080)
    
    print("🧪 Testing web dashboard...")
    print("📱 Open browser: http://localhost:8080")
    print("⌨️  Press Ctrl+C to stop")
    
    try:
        # Simulate processing
        for i in range(1000):
            # Fake frame
            test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.putText(test_frame, f"Frame {i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            # Fake detections  
            fake_detections = [{'class': 'person', 'confidence': 0.85}] if i % 10 == 0 else []
            
            # Process
            dashboard_system.process_frame(test_frame, fake_detections)
            
            if i % 20 == 0:
                dashboard_system.log(f"Processing frame {i}")
                
            if i % 30 == 0:
                dashboard_system.audio_command_sent(f"Test command {i//30}")
            
            time.sleep(0.033)  # ~30 FPS
            
    except KeyboardInterrupt:
        print("\n✅ Web dashboard test completed")