from typing import Union, Optional
import cv2
import numpy as np
from datetime import datetime
import base64
from ...utils.gstreamer import GStreamerPipeline
from ...core.exceptions import CameraError
from ...core.config import CameraConfig
import logging

logger = logging.getLogger('camera.models')

class CameraSource:
    """攝像頭來源類型"""
    LOCAL = "local"
    HLS = "hls"
    RTSP = "rtsp"
    
    @staticmethod
    def detect_source_type(source: Union[int, str]) -> str:
        """檢測攝像頭來源類型"""
        if isinstance(source, int):
            return CameraSource.LOCAL
        elif isinstance(source, str):
            if source.endswith('.m3u8'):
                return CameraSource.HLS
            elif source.startswith('rtsp://'):
                return CameraSource.RTSP
        raise CameraError(f"Unsupported camera source: {source}")

class Camera:
    """攝像頭類，支持本地攝像頭和網絡串流"""
    def __init__(self, config: CameraConfig):
        self._config = config
        self._source = config.source
        self._source_type = CameraSource.detect_source_type(config.source)
        self._cap = None
        self._gst_pipeline = None
        self._is_running = False
        
    def start(self) -> bool:
        """啟動攝像頭"""
        try:
            if self._source_type == CameraSource.LOCAL:
                self._cap = cv2.VideoCapture(self._source)
                if not self._cap.isOpened():
                    raise CameraError(f"Failed to open local camera {self._source}")
                    
                # 設置攝像頭參數
                if self._config.width:
                    self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
                if self._config.height:
                    self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
                if self._config.fps:
                    self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)
                    
            else:
                self._gst_pipeline = GStreamerPipeline()
                if not self._gst_pipeline.create_pipeline(self._source):
                    raise CameraError(f"Failed to create pipeline for {self._source}")
                if not self._gst_pipeline.start():
                    raise CameraError(f"Failed to start pipeline for {self._source}")
                    
            self._is_running = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {str(e)}")
            self.end()
            return False
            
    def end(self):
        """停止攝像頭"""
        self._is_running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._gst_pipeline:
            self._gst_pipeline.stop()
            self._gst_pipeline = None
            
    def capture(self) -> Optional[dict]:
        """捕獲一幀"""
        if not self._is_running:
            return None
            
        try:
            frame = None
            if self._source_type == CameraSource.LOCAL:
                ret, frame = self._cap.read()
                if not ret:
                    raise CameraError("Failed to capture frame from local camera")
            else:
                frame = self._gst_pipeline.get_frame()
                if frame is None:
                    raise CameraError("Failed to get frame from pipeline")
                    
            # 檢查串流狀態
            if self._gst_pipeline:
                status, message = self._gst_pipeline.check_pipeline_status()
                if not status:
                    raise CameraError(f"Pipeline error: {message}")
                    
            # 轉換為base64
            _, buffer = cv2.imencode('.jpg', frame, 
                                   [cv2.IMWRITE_JPEG_QUALITY, self._config.jpeg_quality])
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            return {
                'frame': frame,  # 原始幀用於YOLO處理
                'encoded_frame': jpg_as_text,  # 編碼後的幀用於傳輸
                'timestamp': datetime.now().isoformat(),
                'source_type': self._source_type,
                'source': str(self._source),
                'frame_size': frame.shape[:2]
            }
            
        except Exception as e:
            logger.error(f"Error capturing frame: {str(e)}")
            return None
            
    @property
    def is_running(self) -> bool:
        return self._is_running
        
    def __del__(self):
        """確保資源被正確釋放"""
        self.end() 