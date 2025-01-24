from typing import List, Dict, Any
import numpy as np
from threading import Lock
from ultralytics import YOLO
from ...core.exceptions import YOLOError
from ...core.config import YOLOConfig
import logging

logger = logging.getLogger('yolo.models')

class YOLOProcessor:
    """線程安全的YOLO處理器"""
    def __init__(self, config: YOLOConfig):
        self._config = config
        self._model = None
        self._lock = Lock()
        
    def _ensure_model(self):
        """確保模型已加載（線程安全）"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        self._model = YOLO(self._config.model_path)
                    except Exception as e:
                        raise YOLOError(f"Failed to load YOLO model: {str(e)}")
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """處理單一幀（線程安全）"""
        try:
            self._ensure_model()
            
            # 使用模型進行預測
            results = self._model.predict(
                frame,
                conf=self._config.confidence_threshold,
                verbose=False
            )
            
            # 提取檢測結果
            detections = []
            if len(results) > 0:
                result = results[0]
                for box, conf, cls in zip(result.boxes.xyxy, 
                                        result.boxes.conf, 
                                        result.boxes.cls):
                    if conf >= self._config.confidence_threshold:
                        detections.append({
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'class': int(cls),
                            'class_name': result.names[int(cls)]
                        })
            
            return {
                'detections': detections,
                'processed': True,
                'model_name': self._config.model_path,
                'confidence_threshold': self._config.confidence_threshold
            }
            
        except Exception as e:
            logger.error(f"Error processing frame with YOLO: {str(e)}")
            raise YOLOError(f"YOLO processing error: {str(e)}")

class YOLOProcessorPool:
    """YOLO處理器池，管理多個處理器實例"""
    def __init__(self, config: YOLOConfig):
        self._config = config
        self._processors = [
            YOLOProcessor(config) 
            for _ in range(config.num_threads)
        ]
        self._current = 0
        self._lock = Lock()
        
    def get_processor(self) -> YOLOProcessor:
        """以輪詢方式獲取處理器"""
        with self._lock:
            processor = self._processors[self._current]
            self._current = (self._current + 1) % len(self._processors)
            return processor 