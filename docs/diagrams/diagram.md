src/
├── main.py
├── __init__.py
├── yolo12n.pt
├── core/
│   ├── __init__.py
│   ├── observer.py
│   ├── hardware/
│   │   ├── __init__.py
│   │   └── device_manager.py
│   ├── audio/
│   │   ├── __init__.py
│   │   └── audio_system.py
│   ├── imu/
│   │   ├── __init__.py
│   │   └── motion_detector.py
│   ├── navigation/
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   └── coordinator.py
│   └── vision/
│       ├── __init__.py
│       ├── depth_estimator.py
│       ├── detected_object.py
│       ├── image_enhancer.py
│       ├── object_tracker.py
│       └── yolo_proccesor.py
├── presentation/
│   ├── __init__.py
│   ├── dashboards/
│   │   ├── __init__.py
│   │   ├── opencv_dashboard.py
│   │   └── rerun_dashboard.py
│   └── renderers/
│       ├── __init__.py
│       └── frame_renderer.py
└── utils/
    ├── __init__.py
    ├── config.py
    └── ctrl_handler.py