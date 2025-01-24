import asyncio
import logging
import argparse
from pathlib import Path
from core.config import Config
from blueprints.camera.models import Camera
from blueprints.yolo.models import YOLOProcessorPool
from blueprints.stream.models import StreamManager
from utils.metrics import MetricsCollector
from utils.resource_manager import ResourceManager

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sender.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('main')

class VideoStreamService:
    """視頻串流服務"""
    def __init__(self, config: Config):
        self.config = config
        self.cameras = {}
        self.yolo_pool = None
        self.stream_manager = None
        self.metrics = MetricsCollector()
        self.resource_manager = ResourceManager()
        
    async def setup(self):
        """初始化服務"""
        # 初始化YOLO處理器池
        self.yolo_pool = YOLOProcessorPool(self.config.yolo)
        
        # 初始化所有已啟用的攝像頭
        for cam_id, cam_config in self.config.cameras.items():
            if cam_config.enabled:
                camera = Camera(cam_config)
                if await self._init_camera(camera, cam_id):
                    self.cameras[cam_id] = camera
                    
        if not self.cameras:
            raise RuntimeError("No cameras available")
            
        # 初始化串流管理器
        self.stream_manager = StreamManager(
            self.config.stream,
            self.cameras,
            self.yolo_pool
        )
        
    async def _init_camera(self, camera: Camera, cam_id: str) -> bool:
        """初始化單個攝像頭"""
        try:
            if not camera.start():
                logger.error(f"Failed to start camera {cam_id}")
                return False
            logger.info(f"Camera {cam_id} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing camera {cam_id}: {str(e)}")
            return False
            
    async def run(self):
        """運行服務"""
        try:
            await self.setup()
            
            # 啟動資源監控
            asyncio.create_task(self._monitor_resources())
            
            # 啟動串流服務
            logger.info("Starting video stream service...")
            await self.stream_manager.start()
            
        except Exception as e:
            logger.error(f"Error running service: {str(e)}")
        finally:
            await self.cleanup()
            
    async def _monitor_resources(self):
        """監控系統資源"""
        while True:
            try:
                self.resource_manager.log_resource_usage()
                self.metrics.log_metrics()
                await asyncio.sleep(60)  # 每分鐘記錄一次
            except Exception as e:
                logger.error(f"Error monitoring resources: {str(e)}")
                
    async def cleanup(self):
        """清理資源"""
        logger.info("Cleaning up resources...")
        if self.stream_manager:
            await self.stream_manager.stop()
        for camera in self.cameras.values():
            camera.end()

async def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='Video Stream Service')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    
    args = parser.parse_args()
    config_path = Path(args.config)
    
    try:
        # 加載配置
        config = Config.load_from_file(str(config_path))
        
        # 創建並運行服務
        service = VideoStreamService(config)
        await service.run()
        
    except KeyboardInterrupt:
        logger.info("Received stop signal")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        
if __name__ == '__main__':
    asyncio.run(main())
