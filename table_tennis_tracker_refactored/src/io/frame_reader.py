"""
影像讀取器
負責從攝像頭或影片文件讀取影像幀
"""

import cv2
import time
import threading
import queue
from typing import Tuple, Optional
from ..core.config import CameraConfig, IOConfig


class FrameReader:
    """影像讀取器"""
    
    def __init__(self, video_source, camera_config: CameraConfig, 
                 io_config: IOConfig, use_video_file: bool = False):
        self.video_source = video_source
        self.camera_config = camera_config
        self.io_config = io_config
        self.use_video_file = use_video_file
        
        # 初始化視頻捕獲
        self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_AVFOUNDATION)
        self._configure_capture()
        
        # 線程和隊列
        self.frame_queue = queue.Queue(maxsize=io_config.frame_queue_size)
        self.running = False
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        
        # 獲取實際參數
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 處理無效的FPS值
        if not self.use_video_file and (self.actual_fps <= 0 or self.actual_fps > 1000):
            self.actual_fps = self.camera_config.target_fps
    
    def _configure_capture(self) -> None:
        """配置視頻捕獲參數"""
        if not self.use_video_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.camera_config.target_fps)
            
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {self.video_source}")
    
    def _read_frames(self) -> None:
        """在後台線程中讀取幀"""
        while self.running:
            if not self.frame_queue.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.running = False
                    self.frame_queue.put((False, None))
                    break
                self.frame_queue.put((True, frame))
            else:
                # 隊列滿時稍作等待
                time.sleep(1.0 / (self.camera_config.target_fps * 2))
    
    def start(self) -> None:
        """開始讀取幀"""
        self.running = True
        self.thread.start()
    
    def read(self) -> Tuple[bool, Optional[object]]:
        """讀取一幀"""
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return False, None
    
    def stop(self) -> None:
        """停止讀取幀"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap.isOpened():
            self.cap.release()
    
    def get_properties(self) -> Tuple[float, int, int]:
        """獲取視頻屬性"""
        return self.actual_fps, self.frame_width, self.frame_height
    
    def is_opened(self) -> bool:
        """檢查是否成功打開"""
        return self.cap.isOpened()