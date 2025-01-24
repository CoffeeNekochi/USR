import asyncio
import json
import logging
from typing import Dict, Optional
from datetime import datetime
from ...core.exceptions import StreamError
from ...core.config import StreamConfig
from ..camera.models import Camera
from ..yolo.models import YOLOProcessorPool
import websockets

logger = logging.getLogger('stream.models')

class StreamManager:
    """串流管理器，處理視頻串流和YOLO處理"""
    def __init__(self, config: StreamConfig, camera: Camera, yolo_pool: YOLOProcessorPool):
        self._config = config
        self._camera = camera
        self._yolo_pool = yolo_pool
        self._server = None
        self._is_running = False
        self._last_frame_time = None
        
    async def start(self):
        """啟動串流服務器"""
        try:
            if not self._camera.is_running:
                if not self._camera.start():
                    raise StreamError("Failed to start camera")
                    
            self._server = await websockets.serve(
                self._handle_client,
                self._config.host,
                self._config.port
            )
            
            self._is_running = True
            logger.info(f"Stream server started on {self._config.host}:{self._config.port}")
            
            await self._server.wait_closed()
            
        except Exception as e:
            logger.error(f"Failed to start stream server: {str(e)}")
            raise StreamError(f"Stream server error: {str(e)}")
            
    async def stop(self):
        """停止串流服務器"""
        self._is_running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._camera.end()
        
    async def _handle_client(self, websocket, path):
        """處理WebSocket客戶端連接"""
        client_info = websocket.remote_address
        logger.info(f"New client connected: {client_info}")
        
        try:
            while self._is_running:
                frame_data = await self._process_frame()
                if frame_data:
                    await self._send_frame(websocket, frame_data)
                await asyncio.sleep(1/30)  # 控制幀率
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_info}")
        except Exception as e:
            logger.error(f"Error handling client {client_info}: {str(e)}")
            
    async def _process_frame(self) -> Optional[Dict]:
        """處理一幀"""
        try:
            # 捕獲幀
            frame_data = self._camera.capture()
            if not frame_data:
                return None
                
            # 獲取YOLO處理器並處理幀
            processor = self._yolo_pool.get_processor()
            yolo_results = processor.process_frame(frame_data['frame'])
            
            # 合併結果
            frame_data.update(yolo_results)
            frame_data['timestamp'] = datetime.now().isoformat()
            
            return frame_data
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return None
            
    async def _send_frame(self, websocket, frame_data: Dict):
        """發送處理後的幀"""
        try:
            await websocket.send(json.dumps(frame_data))
        except Exception as e:
            logger.error(f"Error sending frame: {str(e)}")
            raise StreamError(f"Failed to send frame: {str(e)}")
            
    @property
    def is_running(self) -> bool:
        return self._is_running 