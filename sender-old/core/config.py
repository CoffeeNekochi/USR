from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import yaml
import os

@dataclass
class CameraConfig:
    """攝像頭配置"""
    source: Union[int, str]
    name: str  # 攝像頭識別名稱
    fps: int = 30
    width: Optional[int] = None
    height: Optional[int] = None
    buffer_size: int = 1
    enabled: bool = True  # 是否啟用

@dataclass
class YOLOConfig:
    """YOLO配置"""
    model_path: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    num_threads: int = 4
    device: str = "cuda"

@dataclass
class StreamConfig:
    """串流配置"""
    host: str = "0.0.0.0"
    port: int = 5000
    buffer_size: int = 30
    jpeg_quality: int = 95

@dataclass
class Config:
    """全局配置"""
    cameras: Dict[str, CameraConfig] = field(default_factory=dict)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    debug: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'Config':
        """從YAML文件加載配置"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            
        # 創建配置實例
        config = cls()
        
        # 加載攝像頭配置
        if 'cameras' in config_data:
            for cam_name, cam_config in config_data['cameras'].items():
                config.cameras[cam_name] = CameraConfig(
                    name=cam_name,
                    **cam_config
                )
                
        # 加載YOLO配置
        if 'yolo' in config_data:
            config.yolo = YOLOConfig(**config_data['yolo'])
            
        # 加載串流配置
        if 'stream' in config_data:
            config.stream = StreamConfig(**config_data['stream'])
            
        # 加載其他配置
        if 'debug' in config_data:
            config.debug = config_data['debug']
        if 'log_level' in config_data:
            config.log_level = config_data['log_level']
            
        return config
        
    def save_to_file(self, config_path: str):
        """保存配置到YAML文件"""
        config_data = {
            'cameras': {
                name: {
                    'source': cam.source,
                    'fps': cam.fps,
                    'width': cam.width,
                    'height': cam.height,
                    'buffer_size': cam.buffer_size,
                    'enabled': cam.enabled
                }
                for name, cam in self.cameras.items()
            },
            'yolo': {
                'model_path': self.yolo.model_path,
                'confidence_threshold': self.yolo.confidence_threshold,
                'num_threads': self.yolo.num_threads,
                'device': self.yolo.device
            },
            'stream': {
                'host': self.stream.host,
                'port': self.stream.port,
                'buffer_size': self.stream.buffer_size,
                'jpeg_quality': self.stream.jpeg_quality
            },
            'debug': self.debug,
            'log_level': self.log_level
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_data, f, allow_unicode=True) 