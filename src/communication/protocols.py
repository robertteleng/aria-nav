#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Communication Protocols - TFM Navigation System
Definición de contratos de comunicación entre Mac y Jetson

Arquitectura:
Mac (Aria SDK) → FrameMessage → Jetson (Processing) 
Mac (Display) ← ProcessedMessage ← Jetson (Dashboard Gen)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time
import json
import base64
import cv2


# =================================================================
# IMAGE (DE)CODING HELPERS
# =================================================================

def _encode_image(image: np.ndarray, codec: str, quality: int = 85) -> bytes:
    """
    Encode an image using the specified codec.
    Supported: 'jpg'/'jpeg' (lossy), 'png' (lossless).
    """
    if codec.lower() in ("jpg", "jpeg"):
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    elif codec.lower() == "png":
        ok, buf = cv2.imencode(".png", image)
    else:
        raise MessageSerializationError(f"Unsupported codec: {codec}")
    if not ok or buf is None:
        raise MessageSerializationError(f"Image encode failed for codec={codec}")
    return buf.tobytes()


def _decode_image(data: bytes) -> np.ndarray:
    """Decode bytes into a BGR uint8 image (contiguous)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise MessageSerializationError("Image decode failed")
    return np.ascontiguousarray(img)


# =================================================================
# CORE MESSAGE STRUCTURES
# =================================================================

@dataclass
class FrameMessage:
    """
    Mensaje para envío de frames Mac → Jetson
    
    Attributes:
        timestamp: Unix timestamp de captura
        camera_id: "rgb", "slam1", "slam2" 
        frame_data: Frame como numpy array
        frame_shape: Dimensiones del frame (height, width, channels)
        sequence_id: ID secuencial para tracking
        metadata: Info adicional opcional
    """
    timestamp: float
    camera_id: str
    frame_data: np.ndarray
    frame_shape: tuple
    sequence_id: int
    metadata: Optional[Dict[str, Any]] = None

    def to_bytes(self) -> bytes:
        """Serializar mensaje a bytes para envío por socket"""
        codec = getattr(CommunicationConfig, "DEFAULT_FRAME_CODEC", "jpg")
        frame_bytes = _encode_image(self.frame_data, codec, getattr(CommunicationConfig, "JPEG_QUALITY", 85))
        
        # Crear header con metadata
        header = {
            'timestamp': self.timestamp,
            'camera_id': self.camera_id,
            'frame_shape': self.frame_shape,
            'sequence_id': self.sequence_id,
            'frame_size': len(frame_bytes),
            'codec': codec,
            'metadata': self.metadata or {}
        }
        
        # Combinar header + frame
        header_json = json.dumps(header).encode('utf-8')
        header_size = len(header_json).to_bytes(4, byteorder='big')
        return header_size + header_json + frame_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> 'FrameMessage':
        """Deserializar bytes a FrameMessage"""
        # Leer tamaño del header
        header_size = int.from_bytes(data[:4], byteorder='big')
        
        # Leer header
        header_json = data[4:4+header_size]
        header = json.loads(header_json.decode('utf-8'))
        
        # Leer frame data
        frame_bytes = data[4+header_size:4+header_size+header['frame_size']]
        
        # Decode frame (codec stored for future extensions; imdecode auto-detects container)
        frame_data = _decode_image(frame_bytes)
        
        return cls(
            timestamp=header['timestamp'],
            camera_id=header['camera_id'],
            frame_data=frame_data,
            frame_shape=tuple(header['frame_shape']),
            sequence_id=header['sequence_id'],
            metadata=header.get('metadata')
        )


@dataclass
class ProcessedMessage:
    """
    Mensaje de respuesta Jetson → Mac con frame procesado
    
    Attributes:
        timestamp: Timestamp de procesamiento
        sequence_id: ID que corresponde al FrameMessage original
        dashboard_frame: Frame con overlays y anotaciones
        detections: Lista de objetos detectados
        audio_command: Comando de audio generado (opcional)
        performance_metrics: Métricas del procesamiento
        processing_time_ms: Tiempo de procesamiento en milisegundos
    """
    timestamp: float
    sequence_id: int
    dashboard_frame: np.ndarray
    detections: List[Dict[str, Any]]
    audio_command: Optional[str] = None
    performance_metrics: Optional[Dict[str, float]] = None
    processing_time_ms: Optional[float] = None

    def to_bytes(self) -> bytes:
        """Serializar mensaje procesado a bytes"""
        codec = getattr(CommunicationConfig, "DEFAULT_PROCESSED_CODEC", "jpg")
        frame_bytes = _encode_image(self.dashboard_frame, codec, getattr(CommunicationConfig, "JPEG_QUALITY", 85))
        
        # Crear header
        header = {
            'timestamp': self.timestamp,
            'sequence_id': self.sequence_id,
            'frame_size': len(frame_bytes),
            'codec': codec,
            'detections': self.detections,
            'audio_command': self.audio_command,
            'performance_metrics': self.performance_metrics or {},
            'processing_time_ms': self.processing_time_ms
        }
        
        header_json = json.dumps(header).encode('utf-8')
        header_size = len(header_json).to_bytes(4, byteorder='big')
        return header_size + header_json + frame_bytes

    @classmethod
    def from_bytes(cls, data: bytes) -> 'ProcessedMessage':
        """Deserializar bytes a ProcessedMessage"""
        header_size = int.from_bytes(data[:4], byteorder='big')
        header_json = data[4:4+header_size]
        header = json.loads(header_json.decode('utf-8'))
        
        frame_bytes = data[4+header_size:4+header_size+header['frame_size']]
        dashboard_frame = _decode_image(frame_bytes)
        
        return cls(
            timestamp=header['timestamp'],
            sequence_id=header['sequence_id'],
            dashboard_frame=dashboard_frame,
            detections=header['detections'],
            audio_command=header.get('audio_command'),
            performance_metrics=header.get('performance_metrics'),
            processing_time_ms=header.get('processing_time_ms')
        )


# =================================================================
# COMMUNICATION CONFIGURATION
# =================================================================

class CommunicationConfig:
    """Configuración de comunicación entre Mac y Jetson"""
    
    # Network configuration
    JETSON_IP = "192.168.8.204"
    FRAME_PORT = 5555       # Puerto para envío frames Mac → Jetson
    DASHBOARD_PORT = 5556   # Puerto para dashboard Jetson → Mac
    
    # Message configuration
    MAX_MESSAGE_SIZE = 10 * 1024 * 1024  # 10MB max por mensaje
    SOCKET_TIMEOUT = 5.0  # 5 segundos timeout
    RECONNECT_DELAY = 1.0  # 1 segundo entre reintentos conexión
    
    # Performance / encoding configuration
    MAX_FRAME_QUEUE = 3     # Buffer máximo frames en cola
    JPEG_QUALITY = 85       # Calidad compresión (balance size/quality)
    MAX_FPS = 30            # FPS máximo objetivo

    # Default codecs (prod = jpg; tests can override to 'png' lossless)
    DEFAULT_FRAME_CODEC = "jpg"
    DEFAULT_PROCESSED_CODEC = "jpg"

    # Error handling
    MAX_RETRIES = 3         # Reintentos máximos para envío
    HEARTBEAT_INTERVAL = 10.0  # Segundos entre heartbeats


# =================================================================
# MESSAGE UTILITIES
# =================================================================

class MessageUtils:
    """Utilidades para manejo de mensajes"""
    
    @staticmethod
    def create_frame_message(frame: np.ndarray, camera_id: str = "rgb", 
                           metadata: Optional[Dict] = None) -> FrameMessage:
        """Factory method para crear FrameMessage"""
        return FrameMessage(
            timestamp=time.time(),
            camera_id=camera_id,
            frame_data=frame,
            frame_shape=frame.shape,
            sequence_id=int(time.time() * 1000) % 1000000,  # ID basado en timestamp
            metadata=metadata
        )
    
    @staticmethod
    def create_processed_message(dashboard_frame: np.ndarray,
                               detections: List[Dict],
                               original_sequence_id: int,
                               processing_start_time: float,
                               audio_command: Optional[str] = None) -> ProcessedMessage:
        """Factory method para crear ProcessedMessage"""
        processing_time = (time.time() - processing_start_time) * 1000
        
        return ProcessedMessage(
            timestamp=time.time(),
            sequence_id=original_sequence_id,
            dashboard_frame=dashboard_frame,
            detections=detections,
            audio_command=audio_command,
            processing_time_ms=processing_time,
            performance_metrics={
                'fps_estimate': 1000.0 / processing_time if processing_time > 0 else 0,
                'frame_size_kb': dashboard_frame.nbytes / 1024
            }
        )

    @staticmethod
    def validate_frame_message(msg: FrameMessage) -> bool:
        """Validar integridad de FrameMessage"""
        try:
            return (
                msg.frame_data is not None and
                msg.frame_data.size > 0 and
                msg.camera_id in ['rgb', 'slam1', 'slam2'] and
                msg.timestamp > 0 and
                msg.frame_shape == msg.frame_data.shape
            )
        except Exception:
            return False

    @staticmethod
    def validate_processed_message(msg: ProcessedMessage) -> bool:
        """Validar integridad de ProcessedMessage"""
        try:
            return (
                msg.dashboard_frame is not None and
                msg.dashboard_frame.size > 0 and
                msg.sequence_id >= 0 and
                msg.timestamp > 0 and
                isinstance(msg.detections, list)
            )
        except Exception:
            return False


# =================================================================
# ERROR HANDLING
# =================================================================

class CommunicationError(Exception):
    """Base exception para errores de comunicación"""
    pass

class MessageSerializationError(CommunicationError):
    """Error durante serialización/deserialización"""
    pass

class NetworkError(CommunicationError):
    """Error de red/conectividad"""
    pass

class MessageValidationError(CommunicationError):
    """Error de validación de mensaje"""
    pass


# =================================================================
# TESTING UTILITIES
# =================================================================

def test_message_serialization():
    """Test básico de serialización/deserialización"""
    print("Testing message serialization...")

    # Fuerza lossless en test para igualdad byte-a-byte
    prev_frame_codec = getattr(CommunicationConfig, "DEFAULT_FRAME_CODEC", "jpg")
    prev_proc_codec = getattr(CommunicationConfig, "DEFAULT_PROCESSED_CODEC", "jpg")
    try:
        CommunicationConfig.DEFAULT_FRAME_CODEC = "png"
        CommunicationConfig.DEFAULT_PROCESSED_CODEC = "png"

        # Test FrameMessage
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame_msg = MessageUtils.create_frame_message(
            test_frame, 
            camera_id="rgb",
            metadata={"test": True}
        )
        
        # Serialize and deserialize
        frame_bytes = frame_msg.to_bytes()
        reconstructed = FrameMessage.from_bytes(frame_bytes)
        
        assert frame_msg.camera_id == reconstructed.camera_id
        assert frame_msg.sequence_id == reconstructed.sequence_id
        assert np.array_equal(frame_msg.frame_data, reconstructed.frame_data), "Frame data mismatch (lossless expected)"
        print("✅ FrameMessage serialization test passed")
        
        # Test ProcessedMessage
        detections = [{"name": "person", "bbox": [100, 100, 200, 200]}]
        processed_msg = MessageUtils.create_processed_message(
            test_frame,
            detections,
            frame_msg.sequence_id,
            time.time() - 0.1,  # 100ms processing time
            "persona al centro"
        )
        
        processed_bytes = processed_msg.to_bytes()
        reconstructed_processed = ProcessedMessage.from_bytes(processed_bytes)
        
        assert processed_msg.sequence_id == reconstructed_processed.sequence_id
        assert processed_msg.detections == reconstructed_processed.detections
        assert processed_msg.audio_command == reconstructed_processed.audio_command
        assert np.array_equal(processed_msg.dashboard_frame, reconstructed_processed.dashboard_frame), "Dashboard frame mismatch (lossless expected)"
        print("✅ ProcessedMessage serialization test passed")
        
        print(f"Frame message size: {len(frame_bytes)} bytes")
        print(f"Processed message size: {len(processed_bytes)} bytes")
    finally:
        # Restaurar codecs por defecto (producción)
        CommunicationConfig.DEFAULT_FRAME_CODEC = prev_frame_codec
        CommunicationConfig.DEFAULT_PROCESSED_CODEC = prev_proc_codec


if __name__ == "__main__":
    test_message_serialization()
    print("\n🎯 Protocols defined and tested successfully!")
    print(f"Communication configuration:")
    print(f"  - Jetson IP: {CommunicationConfig.JETSON_IP}")
    print(f"  - Frame port: {CommunicationConfig.FRAME_PORT}")
    print(f"  - Dashboard port: {CommunicationConfig.DASHBOARD_PORT}")
    print(f"  - Max message size: {CommunicationConfig.MAX_MESSAGE_SIZE // 1024 // 1024}MB")