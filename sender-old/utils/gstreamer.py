import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
from typing import Optional, Tuple
import logging
from ..core.exceptions import GStreamerError

logger = logging.getLogger('utils.gstreamer')

class GStreamerPipeline:
    """GStreamer管道處理類"""
    
    PIPELINE_TEMPLATES = {
        'hls': (
            'playbin uri={uri} '
            '! decodebin '
            '! videoconvert '
            '! video/x-raw,format=BGR '
            '! appsink name=sink emit-signals=True sync=False'
        ),
        'rtsp': (
            'rtspsrc location={uri} latency=0 '
            '! rtph264depay '
            '! h264parse '
            '! avdec_h264 '
            '! videoconvert '
            '! video/x-raw,format=BGR '
            '! appsink name=sink emit-signals=True sync=False'
        )
    }
    
    def __init__(self):
        """初始化GStreamer"""
        try:
            Gst.init(None)
        except Exception as e:
            raise GStreamerError(f"Failed to initialize GStreamer: {str(e)}")
            
        self.pipeline = None
        self.bus = None
        self.appsink = None
        
    def create_pipeline(self, uri: str) -> bool:
        """創建GStreamer管道"""
        try:
            # 選擇合適的管道模板
            if uri.endswith('.m3u8'):
                template = self.PIPELINE_TEMPLATES['hls']
            elif uri.startswith('rtsp://'):
                template = self.PIPELINE_TEMPLATES['rtsp']
            else:
                raise GStreamerError(f"Unsupported URI format: {uri}")
                
            # 創建管道
            pipeline_str = template.format(uri=uri)
            self.pipeline = Gst.parse_launch(pipeline_str)
            
            # 獲取必要的元素
            self.bus = self.pipeline.get_bus()
            self.appsink = self.pipeline.get_by_name('sink')
            
            if not self.appsink:
                raise GStreamerError("Failed to create appsink element")
                
            return True
            
        except Exception as e:
            raise GStreamerError(f"Failed to create pipeline: {str(e)}")
            
    def start(self) -> bool:
        """啟動管道"""
        if not self.pipeline:
            raise GStreamerError("Pipeline not created")
            
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise GStreamerError("Failed to start pipeline")
            
        return True
        
    def stop(self):
        """停止管道"""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            
    def get_frame(self) -> Optional[np.ndarray]:
        """獲取當前幀"""
        if not self.appsink:
            raise GStreamerError("Pipeline not properly initialized")
            
        # 嘗試獲取樣本
        sample = self.appsink.try_pull_sample(Gst.SECOND)
        if not sample:
            return None
            
        # 獲取buffer和caps
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        
        if not caps:
            raise GStreamerError("No caps in sample")
            
        # 獲取幀的尺寸
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')
        
        # 映射buffer
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            raise GStreamerError("Failed to map buffer")
            
        try:
            # 創建numpy數組
            frame = np.ndarray(
                (height, width, 3),
                dtype=np.uint8,
                buffer=map_info.data
            ).copy()
            return frame
            
        finally:
            buffer.unmap(map_info)
            
    def check_pipeline_status(self) -> Tuple[bool, str]:
        """檢查管道狀態"""
        if not self.bus:
            return False, "No pipeline bus"
            
        # 檢查消息
        msg = self.bus.pop_filtered(
            Gst.MessageType.ERROR | Gst.MessageType.EOS | 
            Gst.MessageType.STATE_CHANGED | Gst.MessageType.STREAM_STATUS
        )
        
        if msg:
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                return False, f"Pipeline error: {err.message}"
            elif msg.type == Gst.MessageType.EOS:
                return False, "End of stream"
                
        return True, "Pipeline running"
        
    def __del__(self):
        """清理資源"""
        self.stop() 