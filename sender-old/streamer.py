import asyncio
import json
import logging
import websockets
from typing import Dict, List, Union, Optional
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from camera import Camera
from threading import Lock
from ultralytics import YOLO
import numpy as np

logger = logging.getLogger('streamer')

class YOLOProcessor:
    """線程安全的YOLO處理器"""
    def __init__(self, model_path: str):
        self._model_path = model_path
        self._model = None
        self._lock = Lock()
        
    def _ensure_model(self):
        """確保模型已加載（線程安全）"""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = YOLO(self._model_path)
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """處理單一幀（線程安全）"""
        self._ensure_model()
        results = self._model.predict(frame, verbose=False)
        
        # 提取檢測結果
        detections = []
        if len(results) > 0:
            result = results[0]
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class': int(cls)
                })
        
        return {
            'detections': detections,
            'processed': True
        }

class FrameProcessor:
    """幀處理器，管理多個YOLO實例"""
    def __init__(self, model_path: str, num_threads: int = 4):
        self._processors = [YOLOProcessor(model_path) for _ in range(num_threads)]
        self._current_processor = 0
        self._lock = Lock()
        
    def get_processor(self) -> YOLOProcessor:
        """以輪詢方式獲取處理器"""
        with self._lock:
            processor = self._processors[self._current_processor]
            self._current_processor = (self._current_processor + 1) % len(self._processors)
            return processor

class FrameBuffer:
    """幀緩衝區，用於管理原始幀和處理後的幀"""
    def __init__(self, maxsize: int = 30):
        self.raw_frames = Queue(maxsize)
        self.processed_frames = Queue(maxsize)
        
    def put_raw(self, frame: dict) -> None:
        """添加原始幀，如果隊列滿則丟棄最舊的幀"""
        if self.raw_frames.full():
            self.raw_frames.get()
        self.raw_frames.put(frame)
        
    def put_processed(self, frame: dict) -> None:
        """添加處理後的幀，如果隊列滿則丟棄最舊的幀"""
        if self.processed_frames.full():
            self.processed_frames.get()
        self.processed_frames.put(frame)
        
    def get_latest_processed(self) -> Optional[dict]:
        """獲取最新的處理後幀"""
        if self.processed_frames.empty():
            return None
        return self.processed_frames.get()

class CameraManager:
    """管理多個攝像頭的類"""
    def __init__(self, model_path: str, buffer_size: int = 30, num_threads: int = 4):
        self._cameras: Dict[str, Camera] = {}
        self._frame_buffers: Dict[str, FrameBuffer] = {}
        self._lock = asyncio.Lock()
        self._buffer_size = buffer_size
        self._frame_processor = FrameProcessor(model_path, num_threads)
        self._thread_pool = ThreadPoolExecutor(max_workers=num_threads)
        
    def add_camera(self, camera_id: str, source: Union[int, str]) -> None:
        """添加新的攝像頭"""
        if camera_id in self._cameras:
            raise ValueError(f"Camera ID {camera_id} already exists")
        self._cameras[camera_id] = Camera(source)
        self._frame_buffers[camera_id] = FrameBuffer(self._buffer_size)
        
    async def update_frame(self, camera_id: str) -> None:
        """更新指定攝像頭的幀"""
        if camera_id not in self._cameras:
            return
            
        try:
            frame_data = self._cameras[camera_id].capture()
            self._frame_buffers[camera_id].put_raw(frame_data)
            
            # 在線程池中進行YOLO處理
            processed_frame = await asyncio.get_event_loop().run_in_executor(
                self._thread_pool,
                self._process_frame,
                frame_data
            )
            
            self._frame_buffers[camera_id].put_processed(processed_frame)
        except Exception as e:
            logger.error(f"Camera {camera_id} capture error: {str(e)}")
            
    def _process_frame(self, frame_data: dict) -> dict:
        """在獨立線程中進行YOLO處理"""
        try:
            # 獲取一個可用的處理器
            processor = self._frame_processor.get_processor()
            
            # 處理幀
            frame = frame_data['frame']  # 假設frame_data中包含原始幀數據
            results = processor.process_frame(frame)
            
            # 合併結果
            frame_data.update(results)
            return frame_data
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return frame_data
            
    async def get_all_processed_frames(self) -> dict:
        """獲取所有攝像頭的最新處理後幀"""
        processed_frames = {}
        for camera_id, buffer in self._frame_buffers.items():
            frame = buffer.get_latest_processed()
            if frame:
                processed_frames[camera_id] = frame
        return processed_frames

class VideoStreamer:
    def __init__(self, model_path: str = "yolov8n.pt", buffer_size: int = 30, num_threads: int = 4):
        self._camera_manager = CameraManager(model_path, buffer_size, num_threads)
        self._server = None
        self._host = "0.0.0.0"
        self._port = 5000
        self._capture_tasks = []
        self._processing_tasks = []
        
    async def capture_and_process(self, camera_id: str):
        """持續捕獲和處理單個攝像頭的幀"""
        logger.info(f"Starting capture for camera {camera_id}")
        while True:
            try:
                await self._camera_manager.update_frame(camera_id)
                await asyncio.sleep(1/30)  # 控制捕獲頻率
            except Exception as e:
                logger.error(f"Error capturing from camera {camera_id}: {str(e)}")
                await asyncio.sleep(1)
                
    async def _send_frames(self, websocket):
        """發送所有攝像頭的最新處理後幀"""
        try:
            frames = await self._camera_manager.get_all_processed_frames()
            if frames:
                await websocket.send(json.dumps({
                    'timestamp': frames[next(iter(frames))]['timestamp'],
                    'cameras': frames
                }))
        except Exception as e:
            logger.error(f"Error sending frames: {str(e)}")
            raise
            
    async def start(self):
        """啟動服務器"""
        # 為每個攝像頭創建捕獲任務
        for camera_id in self._camera_manager._cameras.keys():
            task = asyncio.create_task(self.capture_and_process(camera_id))
            self._capture_tasks.append(task)
            
        # 啟動WebSocket服務器
        self._server = await websockets.serve(
            self.handle_client,
            self._host,
            self._port,
        )
        
        logger.info(f"Video streaming server started on {self._host}:{self._port}")
        await self._server.wait_closed()
        
    async def stop(self):
        """停止服務器"""
        # 取消所有捕獲任務
        for task in self._capture_tasks:
            task.cancel()
            
        # 等待任務完成
        await asyncio.gather(*self._capture_tasks, return_exceptions=True)
        
        # 關閉服務器
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            
    @classmethod
    async def run(cls):
        """運行串流服務器"""
        streamer = cls()
        
        # 添加攝像頭示例
        streamer.add_camera('cam1', 0)  # 本地攝像頭
        streamer.add_camera('cam2', "http://example.com/stream1.m3u8")  # IP攝像頭1
        streamer.add_camera('cam3', "http://example.com/stream2.m3u8")  # IP攝像頭2
        
        try:
            await streamer.start()
        except KeyboardInterrupt:
            logger.info("Received stop signal")
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
        finally:
            await streamer.stop()




