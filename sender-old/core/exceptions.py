class SenderError(Exception):
    """基礎異常類"""
    pass

class CameraError(SenderError):
    """攝像頭相關異常"""
    pass

class StreamError(SenderError):
    """串流相關異常"""
    pass

class YOLOError(SenderError):
    """YOLO處理相關異常"""
    pass

class ConfigError(SenderError):
    """配置相關異常"""
    pass

class GStreamerError(SenderError):
    """GStreamer相關異常"""
    pass 