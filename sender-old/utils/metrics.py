import time
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger('utils.metrics')

@dataclass
class PerformanceMetrics:
    """性能指標數據類"""
    fps: float = 0.0
    processing_time: float = 0.0
    frame_count: int = 0
    yolo_time: float = 0.0
    network_time: float = 0.0
    timestamp: str = ""

class MetricsCollector:
    """性能指標收集器"""
    def __init__(self):
        self._start_time = time.time()
        self._frame_count = 0
        self._last_fps_time = self._start_time
        self._current_fps = 0.0
        self._metrics_history: Dict[str, PerformanceMetrics] = {}
        
    def start_operation(self, operation_name: str):
        """開始計時操作"""
        return Timer(operation_name, self)
        
    def update_metrics(self, operation_name: str, duration: float):
        """更新操作指標"""
        if operation_name not in self._metrics_history:
            self._metrics_history[operation_name] = PerformanceMetrics()
            
        metrics = self._metrics_history[operation_name]
        metrics.processing_time = duration
        metrics.timestamp = datetime.now().isoformat()
        
        if operation_name == "frame_processing":
            self._frame_count += 1
            current_time = time.time()
            time_diff = current_time - self._last_fps_time
            
            if time_diff >= 1.0:  # 每秒更新一次FPS
                self._current_fps = self._frame_count / time_diff
                metrics.fps = self._current_fps
                metrics.frame_count = self._frame_count
                
                self._frame_count = 0
                self._last_fps_time = current_time
                
    def get_metrics(self, operation_name: str) -> Optional[PerformanceMetrics]:
        """獲取指定操作的指標"""
        return self._metrics_history.get(operation_name)
        
    def get_current_fps(self) -> float:
        """獲取當前FPS"""
        return self._current_fps
        
    def log_metrics(self):
        """記錄性能指標"""
        for op_name, metrics in self._metrics_history.items():
            logger.info(f"{op_name} metrics: {metrics}")

class Timer:
    """上下文管理器用於計時"""
    def __init__(self, operation_name: str, collector: MetricsCollector):
        self.operation_name = operation_name
        self.collector = collector
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.update_metrics(self.operation_name, duration) 