"""
Professional HTML template for Web Dashboard - No emojis, modern design
"""

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Aria Navigation Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: #0a0e27;
            color: #e4e4e7;
            overflow-x: hidden;
        }
        
        .header { 
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            padding: 1.5rem 2rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            border-bottom: 1px solid rgba(59, 130, 246, 0.3);
        }
        
        .header-content {
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }
        
        .header h1 { 
            font-size: 1.75rem;
            font-weight: 600;
            letter-spacing: -0.025em;
            color: white;
        }
        
        .controls { 
            display: flex;
            gap: 0.75rem;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 1.5rem;
        }
        
        .grid { 
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }
        
        .panel { 
            background: linear-gradient(145deg, #1a1d35, #151828);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
            border: 1px solid rgba(59, 130, 246, 0.1);
            transition: all 0.3s ease;
        }
        
        .panel:hover {
            border-color: rgba(59, 130, 246, 0.3);
            box-shadow: 0 6px 16px rgba(0,0,0,0.5);
        }
        
        .panel-header { 
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid rgba(59, 130, 246, 0.2);
        }
        
        .panel-title { 
            font-size: 1.125rem;
            font-weight: 600;
            color: #3b82f6;
        }
        
        .frame { 
            width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            background: #000;
        }
        
        .frame-half {
            width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            background: #000;
        }
        
        .stats-grid { 
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 1rem;
        }

        .system-metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
        }

        .stat-detail {
            font-size: 0.7rem;
            color: #94a3b8;
            margin-top: 0.35rem;
            letter-spacing: 0.05em;
        }
        
        .stat { 
            background: linear-gradient(145deg, #1e293b, #0f172a);
            padding: 1.25rem;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(59, 130, 246, 0.1);
            transition: all 0.2s ease;
        }
        
        .stat:hover {
            border-color: rgba(59, 130, 246, 0.3);
            transform: translateY(-2px);
        }
        
        .stat-value { 
            font-size: 2rem;
            font-weight: 700;
            color: #3b82f6;
            margin-bottom: 0.5rem;
            font-variant-numeric: tabular-nums;
        }
        
        .stat-label { 
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 500;
        }
        
        .stat.critical .stat-value { color: #ef4444; }
        .stat.warning .stat-value { color: #f59e0b; }
        .stat.success .stat-value { color: #10b981; }
        
        .logs { 
            height: 320px;
            overflow-y: auto;
            background: #000;
            padding: 1rem;
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            border-radius: 8px;
            border: 1px solid rgba(59, 130, 246, 0.1);
            font-size: 0.8125rem;
            line-height: 1.6;
        }
        
        .logs::-webkit-scrollbar { width: 8px; }
        .logs::-webkit-scrollbar-track { background: #0a0e27; }
        .logs::-webkit-scrollbar-thumb { background: #3b82f6; border-radius: 4px; }
        
        .log-entry { margin: 0.25rem 0; padding: 0.125rem 0; }
        .log-info { color: #10b981; }
        .log-warn { color: #f59e0b; }
        .log-error { color: #ef4444; }
        .log-audio { color: #a78bfa; }
        .log-detect { color: #3b82f6; }
        .log-system { color: #6ee7b7; }
        
        button { 
            padding: 0.625rem 1.25rem;
            background: linear-gradient(145deg, #3b82f6, #2563eb);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.875rem;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
        }
        
        button:hover { 
            background: linear-gradient(145deg, #2563eb, #1d4ed8);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(59, 130, 246, 0.4);
        }
        
        button:active { transform: translateY(0); }
        
        .status { 
            padding: 0.5rem 1rem;
            background: #1e293b;
            border-radius: 20px;
            font-size: 0.875rem;
            border: 1px solid rgba(59, 130, 246, 0.2);
            font-weight: 500;
        }
        
        .status.online { 
            background: linear-gradient(145deg, #065f46, #047857);
            border-color: #10b981;
        }
        
        .detection-item {
            background: linear-gradient(145deg, #1e293b, #0f172a);
            margin: 0.5rem 0;
            padding: 0.875rem;
            border-radius: 6px;
            border-left: 3px solid #3b82f6;
            font-size: 0.875rem;
            transition: all 0.2s ease;
        }
        
        .detection-item:hover {
            border-left-color: #60a5fa;
            transform: translateX(4px);
        }
        
        .no-data {
            text-align: center;
            color: #64748b;
            padding: 2rem;
            font-style: italic;
        }
        
        .slam-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }
        
        .slam-item h4 {
            font-size: 0.875rem;
            color: #94a3b8;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }
        
        .slam-events {
            margin-top: 1rem;
            font-size: 0.8125rem;
            color: #cbd5e1;
            background: rgba(0,0,0,0.3);
            padding: 0.75rem;
            border-radius: 6px;
            max-height: 120px;
            overflow-y: auto;
        }
        
        .beep-stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }
        
        .beep-stat-card {
            background: linear-gradient(145deg, #1e293b, #0f172a);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(59, 130, 246, 0.1);
        }
        
        .beep-stat-card.critical { border-left: 3px solid #ef4444; }
        .beep-stat-card.normal { border-left: 3px solid #10b981; }
        
        .beep-stat-header {
            font-size: 0.875rem;
            color: #94a3b8;
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.05em;
        }
        
        .beep-stat-value {
            font-size: 1.75rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        
        .beep-stat-card.critical .beep-stat-value { color: #ef4444; }
        .beep-stat-card.normal .beep-stat-value { color: #10b981; }
        
        .beep-frequency {
            font-size: 0.75rem;
            color: #64748b;
        }
        
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
            .slam-container { grid-template-columns: 1fr; }
            .header-content { flex-direction: column; align-items: stretch; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <h1>Aria Navigation Dashboard</h1>
            <div class="controls">
                <button onclick="location.reload()">Refresh</button>
                <button onclick="toggleAutoRefresh()" id="auto-btn">Start Auto Refresh</button>
                <span class="status" id="status">Status: Loading...</span>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="grid">
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">RGB Camera + Object Detection</span>
                </div>
                <img id="rgb-frame" class="frame" src="/video_feed/rgb" alt="RGB Feed">
            </div>
            
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">Depth Map</span>
                </div>
                <img id="depth-frame" class="frame" src="/video_feed/depth" alt="Depth Feed">
            </div>

            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">Peripheral Vision (SLAM)</span>
                </div>
                <div class="slam-container">
                    <div class="slam-item">
                        <h4>Left Camera</h4>
                        <img id="slam1-frame" class="frame-half" src="/video_feed/slam1" alt="SLAM1">
                    </div>
                    <div class="slam-item">
                        <h4>Right Camera</h4>
                        <img id="slam2-frame" class="frame-half" src="/video_feed/slam2" alt="SLAM2">
                    </div>
                </div>
                <div id="slam-events" class="slam-events">
                    <div>Lateral events will appear here...</div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">System Logs</span>
                </div>
                <div class="logs" id="logs-container">Loading logs...</div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">Performance Metrics</span>
                </div>
                <div class="stats-grid">
                    <div class="stat success">
                        <div class="stat-value" id="fps">0.0</div>
                        <div class="stat-label">FPS</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="frames">0</div>
                        <div class="stat-label">Frames</div>
                    </div>
                    <div class="stat warning">
                        <div class="stat-value" id="detections">0</div>
                        <div class="stat-label">Detections</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="audio">0</div>
                        <div class="stat-label">Audio Alerts</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="slam1-count">0</div>
                        <div class="stat-label">SLAM Left</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="slam2-count">0</div>
                        <div class="stat-label">SLAM Right</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="uptime">0s</div>
                        <div class="stat-label">Uptime</div>
                    </div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">System Metrics</span>
                </div>
                <div class="stats-grid system-metrics-grid">
                    <div class="stat">
                        <div class="stat-value" id="cpu-panel">--%</div>
                        <div class="stat-label">CPU Usage</div>
                        <div class="stat-detail" id="cpu-panel-detail">Resource monitor sampling</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="ram-panel">--/-- MB</div>
                        <div class="stat-label">RAM Usage</div>
                        <div class="stat-detail" id="ram-panel-detail">--% used</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="gpu-panel">N/A</div>
                        <div class="stat-label">GPU Utilization</div>
                        <div class="stat-detail" id="gpu-panel-detail">No GPU detected</div>
                    </div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">Audio Beeps Statistics</span>
                </div>
                <div class="beep-stats-grid">
                    <div class="beep-stat-card critical">
                        <div class="beep-stat-header">Critical Beeps</div>
                        <div class="beep-stat-value" id="critical-beeps">0</div>
                        <div class="beep-frequency" id="critical-freq">Frequency: -- Hz</div>
                    </div>
                    <div class="beep-stat-card normal">
                        <div class="beep-stat-header">Normal Beeps</div>
                        <div class="beep-stat-value" id="normal-beeps">0</div>
                        <div class="beep-frequency" id="normal-freq">Frequency: -- Hz</div>
                    </div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">
                    <span class="panel-title">Recent Detections</span>
                </div>
                <div id="detections-container">
                    <div class="no-data">No recent detections</div>
                </div>
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
                status.textContent = 'Status: Active';
                status.className = 'status online';
                btn.textContent = 'Stop Auto Refresh';
                startAutoRefresh();
            } else {
                status.textContent = 'Status: Paused';
                status.className = 'status';
                btn.textContent = 'Start Auto Refresh';
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
            if (!autoRefresh) return;
            
            fetch('/stats')
                .then(response => response.json())
                .then(data => {
                    console.log('Stats update:', data);
                    
                    document.getElementById('fps').textContent = (data.fps || 0).toFixed(1);
                    document.getElementById('frames').textContent = data.frames_processed || 0;
                    document.getElementById('detections').textContent = data.detections_total || 0;
                    document.getElementById('audio').textContent = data.audio_commands || 0;
                    document.getElementById('slam1-count').textContent = data.slam1_events || 0;
                    document.getElementById('slam2-count').textContent = data.slam2_events || 0;
                    
                    // Beep stats
                    document.getElementById('critical-beeps').textContent = data.critical_beeps || 0;
                    document.getElementById('normal-beeps').textContent = data.normal_beeps || 0;
                    
                    const critFreq = data.critical_frequency || 0;
                    const normFreq = data.normal_frequency || 0;
                    document.getElementById('critical-freq').textContent = 
                        critFreq > 0 ? `Frequency: ${critFreq} Hz` : 'Frequency: -- Hz';
                    document.getElementById('normal-freq').textContent = 
                        normFreq > 0 ? `Frequency: ${normFreq} Hz` : 'Frequency: -- Hz';
                    
                    const eventsBox = document.getElementById('slam-events');
                    if (data.slam_messages && data.slam_messages.length) {
                        eventsBox.innerHTML = data.slam_messages.map(msg => `<div>${msg}</div>`).join('');
                    } else {
                        eventsBox.innerHTML = '<div>No recent SLAM events</div>';
                    }
                    document.getElementById('uptime').textContent = formatUptime(data.uptime || 0);

                    const cpuPct = data.cpu_pct ? data.cpu_pct.toFixed(1) : null;
                    const cpuText = cpuPct ? `${cpuPct}%` : '--%';
                    document.getElementById('cpu-panel').textContent = cpuText;
                    document.getElementById('cpu-panel-detail').textContent = cpuPct ? 'Live ResourceMonitor sample' : 'Resource monitor sampling';

                    const ramUsed = data.ram_used_mb || 0;
                    const ramTotal = data.ram_total_mb || 0;
                    const ramPct = ramTotal ? Math.round((ramUsed / ramTotal) * 100) : 0;
                    const ramText = ramTotal ? `${ramUsed}/${ramTotal} MB` : '--/-- MB';
                    document.getElementById('ram-panel').textContent = ramText;
                    document.getElementById('ram-panel-detail').textContent = ramTotal ? `${ramPct}% used` : '--% used';

                    let gpuText = 'N/A';
                    let gpuDetail = 'GPU not detected';
                    if (data.gpu_present) {
                        const gpuUtil = data.gpu_util_pct || 0;
                        const gpuMemUsed = data.gpu_mem_used_mb || 0;
                        const gpuMemTotal = data.gpu_mem_total_mb || 0;
                        gpuText = `${gpuUtil}% / ${gpuMemUsed}MB`;
                        gpuDetail = gpuMemTotal ? `Total ${gpuMemTotal}MB` : 'Memory snapshot pending';
                    }
                    document.getElementById('gpu-panel').textContent = gpuText;
                    document.getElementById('gpu-panel-detail').textContent = gpuDetail;
                })
                .catch(err => {
                    console.error('Stats error:', err);
                    document.getElementById('fps').textContent = 'ERR';
                });
            
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
                .catch(err => console.error('Logs error:', err));
            
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
                .catch(err => console.error('Detections error:', err));
        }
        
        function formatUptime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            
            if (hours > 0) return `${hours}h ${minutes}m`;
            else if (minutes > 0) return `${minutes}m ${secs}s`;
            else return `${secs}s`;
        }
        
        window.addEventListener('load', () => {
            setTimeout(() => toggleAutoRefresh(), 1000);
        });
    </script>
</body>
</html>
'''
