import psutil
import GPUtil
from typing import Dict, List
import logging
from dataclasses import dataclass

logger = logging.getLogger('utils.resource_manager')

@dataclass
class SystemResources:
    """系統資源使用情況"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_utilization: List[float] = None
    gpu_memory_used: List[float] = None
    disk_usage: float = 0.0

class ResourceManager:
    """資源管理器"""
    def __init__(self):
        self._process = psutil.Process()
        
    def get_system_resources(self) -> SystemResources:
        """獲取系統資源使用情況"""
        try:
            resources = SystemResources()
            
            # CPU使用率
            resources.cpu_percent = psutil.cpu_percent()
            
            # 內存使用率
            memory = psutil.virtual_memory()
            resources.memory_percent = memory.percent
            
            # GPU使用率（如果有GPU）
            try:
                gpus = GPUtil.getGPUs()
                resources.gpu_utilization = [gpu.load * 100 for gpu in gpus]
                resources.gpu_memory_used = [gpu.memoryUsed for gpu in gpus]
            except Exception as e:
                logger.warning(f"Failed to get GPU information: {str(e)}")
                
            # 磁盤使用率
            disk = psutil.disk_usage('/')
            resources.disk_usage = disk.percent
            
            return resources
            
        except Exception as e:
            logger.error(f"Error getting system resources: {str(e)}")
            return SystemResources()
            
    def get_process_resources(self) -> Dict:
        """獲取當前進程的資源使用情況"""
        try:
            return {
                'cpu_percent': self._process.cpu_percent(),
                'memory_percent': self._process.memory_percent(),
                'num_threads': self._process.num_threads(),
                'open_files': len(self._process.open_files()),
                'connections': len(self._process.connections())
            }
        except Exception as e:
            logger.error(f"Error getting process resources: {str(e)}")
            return {}
            
    def check_resources(self) -> bool:
        """檢查資源是否足夠"""
        try:
            resources = self.get_system_resources()
            
            # 檢查CPU使用率
            if resources.cpu_percent > 90:
                logger.warning("High CPU usage detected")
                return False
                
            # 檢查內存使用率
            if resources.memory_percent > 90:
                logger.warning("High memory usage detected")
                return False
                
            # 檢查GPU使用率
            if resources.gpu_utilization:
                for i, util in enumerate(resources.gpu_utilization):
                    if util > 90:
                        logger.warning(f"High GPU {i} usage detected")
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"Error checking resources: {str(e)}")
            return False
            
    def log_resource_usage(self):
        """記錄資源使用情況"""
        try:
            system_resources = self.get_system_resources()
            process_resources = self.get_process_resources()
            
            logger.info("System resources: %s", system_resources)
            logger.info("Process resources: %s", process_resources)
            
        except Exception as e:
            logger.error(f"Error logging resource usage: {str(e)}") 