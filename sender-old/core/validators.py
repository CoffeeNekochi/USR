from typing import Optional
from .config import Config, CameraConfig, YOLOConfig, StreamConfig
from .exceptions import ConfigError

def validate_camera_config(config: CameraConfig) -> Optional[str]:
    """驗證攝像頭配置"""
    if not config.source:
        return "Camera source is required"
        
    if isinstance(config.source, str):
        if not (config.source.endswith('.m3u8') or 
                config.source.startswith('rtsp://')):
            return "Invalid camera source URL format"
            
    if config.fps and config.fps <= 0:
        return "FPS must be positive"
        
    if config.width and config.width <= 0:
        return "Width must be positive"
        
    if config.height and config.height <= 0:
        return "Height must be positive"
        
    return None

def validate_yolo_config(config: YOLOConfig) -> Optional[str]:
    """驗證YOLO配置"""
    if not config.model_path:
        return "Model path is required"
        
    if not config.model_path.endswith('.pt'):
        return "Invalid model file format"
        
    if config.confidence_threshold < 0 or config.confidence_threshold > 1:
        return "Confidence threshold must be between 0 and 1"
        
    if config.num_threads <= 0:
        return "Number of threads must be positive"
        
    if config.device not in ['cuda', 'cpu']:
        return "Device must be either 'cuda' or 'cpu'"
        
    return None

def validate_stream_config(config: StreamConfig) -> Optional[str]:
    """驗證串流配置"""
    if not config.host:
        return "Host is required"
        
    if config.port <= 0 or config.port > 65535:
        return "Port must be between 1 and 65535"
        
    if config.buffer_size <= 0:
        return "Buffer size must be positive"
        
    if config.jpeg_quality <= 0 or config.jpeg_quality > 100:
        return "JPEG quality must be between 1 and 100"
        
    return None

def validate_config(config: Config) -> None:
    """驗證全局配置"""
    # 驗證攝像頭配置
    if error := validate_camera_config(config.camera):
        raise ConfigError(f"Camera configuration error: {error}")
        
    # 驗證YOLO配置
    if error := validate_yolo_config(config.yolo):
        raise ConfigError(f"YOLO configuration error: {error}")
        
    # 驗證串流配置
    if error := validate_stream_config(config.stream):
        raise ConfigError(f"Stream configuration error: {error}")
        
    # 驗證日誌級別
    valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    if config.log_level not in valid_log_levels:
        raise ConfigError(f"Invalid log level. Must be one of {valid_log_levels}") 