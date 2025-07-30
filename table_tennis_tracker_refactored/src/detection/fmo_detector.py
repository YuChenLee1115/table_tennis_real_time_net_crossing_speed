"""
快速移動物體(FMO)偵測器
使用幀差法檢測快速移動的物體
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple
from ..core.config import DetectionConfig


class FMODetector:
    """快速移動物體偵測器"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.prev_frames = deque(maxlen=config.max_prev_frames_fmo)
        
        # 建立形態學核心
        self.opening_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, config.opening_kernel_size_fmo
        )
        self.closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, config.closing_kernel_size_fmo
        )
    
    def preprocess_frame(self, roi_frame: np.ndarray) -> np.ndarray:
        """預處理ROI幀"""
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self.prev_frames.append(blurred)
        return blurred
    
    def detect_motion(self) -> Optional[np.ndarray]:
        """偵測運動物體並返回二值化遮罩"""
        if len(self.prev_frames) < 3:
            return None
        
        # 取最近三幀進行差分
        f1, f2, f3 = self.prev_frames[-3], self.prev_frames[-2], self.prev_frames[-1]
        
        # 計算幀差
        diff1 = cv2.absdiff(f1, f2)
        diff2 = cv2.absdiff(f2, f3)
        motion_mask = cv2.bitwise_and(diff1, diff2)
        
        # 二值化
        try:
            _, thresh_mask = cv2.threshold(
                motion_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        except cv2.error:
            _, thresh_mask = cv2.threshold(
                motion_mask, self.config.threshold_value_fmo, 255, cv2.THRESH_BINARY
            )
        
        # 形態學處理
        if self.config.opening_kernel_size_fmo[0] > 0:
            opened_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, self.opening_kernel)
        else:
            opened_mask = thresh_mask
            
        closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, self.closing_kernel)
        
        return closed_mask
    
    def reset(self) -> None:
        """重置偵測器狀態"""
        self.prev_frames.clear()