"""
快速移動物體(FMO)偵測器
使用多幀差分和自適應背景建模檢測快速移動的物體
針對桌球等高速小目標進行優化
"""

import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple, List
from ..core.config import DetectionConfig
from ..utils.performance_optimizer import get_performance_optimizer


class FMODetector:
    """快速移動物體偵測器 - 專門針對桌球等高速小目標優化
    
    主要特點:
    - 多幀差分提高檢測穩定性
    - 自適應閾值處理不同光照條件
    - 形態學操作去除雜訊並增強小目標
    - 支援M2 Pro MacBook的GPU加速
    """
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        
        # 性能優化器
        self.performance_optimizer = get_performance_optimizer()
        
        # 存儲預處理後的灰階幀，用於多幀差分
        self.prev_frames = deque(maxlen=config.max_prev_frames_fmo)
        
        # 背景模型用於適應光照變化
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=20, varThreshold=50, detectShadows=False
        )
        
        # 自適應閾值歷史，用於動態調整檢測靈敏度
        self.threshold_history = deque(maxlen=10)
        self.adaptive_threshold = config.threshold_value_fmo
        
        # 建立形態學核心 - 使用性能優化器
        self.opening_kernel = self.performance_optimizer.optimize_morphological_operations(
            'opening', config.opening_kernel_size_fmo
        )
        self.closing_kernel = self.performance_optimizer.optimize_morphological_operations(
            'closing', config.closing_kernel_size_fmo
        )
        
        # 額外的小目標增強核心
        self.ball_enhance_kernel = self.performance_optimizer.optimize_morphological_operations(
            'dilate', (3, 3)
        )
        
        # 幀間統計，用於品質評估
        self.frame_stats = {
            'brightness_mean': 0,
            'contrast_std': 0,
            'motion_density': 0
        }
    
    def preprocess_frame(self, roi_frame: np.ndarray) -> np.ndarray:
        """預處理ROI幀 - 增強對比度和減少雜訊
        
        Args:
            roi_frame: 輸入的ROI彩色幀
            
        Returns:
            處理後的灰階幀
        """
        # 轉換為灰階
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # 計算幀統計用於自適應處理
        self.frame_stats['brightness_mean'] = np.mean(gray)
        self.frame_stats['contrast_std'] = np.std(gray)
        
        # 自適應直方圖均化，改善不同光照條件下的檢測
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 高斯模糊去除雜訊，使用性能優化器
        # 對於桌球，使用較小的核心以保持邊緣清晰
        kernel_size = 3 if self.frame_stats['brightness_mean'] > 100 else 5
        blurred = self.performance_optimizer.parallel_gaussian_blur(
            enhanced, (kernel_size, kernel_size), 0
        )
        
        self.prev_frames.append(blurred)
        return blurred
    
    def detect_motion(self) -> Optional[np.ndarray]:
        """偵測運動物體並返回二值化遮罩
        
        使用多種技術提高檢測準確性：
        1. 多幀差分增強運動檢測
        2. 背景建模適應環境變化
        3. 自適應閾值處理不同光照
        4. 形態學操作去雜訊和連接斷點
        
        Returns:
            運動物體的二值化遮罩，白色區域為檢測到的運動物體
        """
        if len(self.prev_frames) < 3:
            return None
        
        current_frame = self.prev_frames[-1]
        
        # 方法1: 多幀差分檢測快速運動
        motion_mask_diff = self._multi_frame_difference()
        
        # 方法2: 背景建模檢測持續運動
        motion_mask_bg = self.background_subtractor.apply(current_frame)
        
        # 結合兩種方法，取並集以捕捉更多運動
        combined_mask = cv2.bitwise_or(motion_mask_diff, motion_mask_bg)
        
        # 自適應二值化，根據幀品質調整閾值
        refined_mask = self._adaptive_thresholding(combined_mask)
        
        # 形態學處理，針對桌球大小優化
        final_mask = self._morphological_processing(refined_mask)
        
        # 更新運動密度統計
        self.frame_stats['motion_density'] = np.sum(final_mask > 0) / final_mask.size
        
        return final_mask
    
    def _multi_frame_difference(self) -> np.ndarray:
        """多幀差分檢測，提高對快速運動物體的敏感性"""
        frames = list(self.prev_frames)
        n_frames = len(frames)
        
        if n_frames < 3:
            return np.zeros_like(frames[0])
        
        # 計算多個幀差並取最大值
        diff_masks = []
        
        # 連續幀差
        for i in range(n_frames - 1):
            diff = cv2.absdiff(frames[i], frames[i + 1])
            diff_masks.append(diff)
        
        # 跨幀差（檢測更快速的運動）
        if n_frames >= 4:
            skip_diff = cv2.absdiff(frames[-4], frames[-1])
            diff_masks.append(skip_diff)
        
        # 取所有差分的最大值
        motion_mask = np.maximum.reduce(diff_masks)
        
        return motion_mask
    
    def _adaptive_thresholding(self, motion_mask: np.ndarray) -> np.ndarray:
        """自適應閾值處理，根據幀品質動態調整"""
        # 計算運動強度統計
        motion_mean = np.mean(motion_mask)
        motion_std = np.std(motion_mask)
        
        # 更新閾值歷史
        current_threshold = motion_mean + 2 * motion_std
        self.threshold_history.append(current_threshold)
        
        # 計算自適應閾值（平滑化）
        if len(self.threshold_history) > 0:
            self.adaptive_threshold = np.median(list(self.threshold_history))
        
        # 根據光照條件調整
        brightness_factor = 1.0
        if self.frame_stats['brightness_mean'] < 80:  # 暗環境
            brightness_factor = 0.8
        elif self.frame_stats['brightness_mean'] > 180:  # 亮環境
            brightness_factor = 1.2
        
        final_threshold = max(10, self.adaptive_threshold * brightness_factor)
        
        # 應用閾值
        _, thresh_mask = cv2.threshold(
            motion_mask, final_threshold, 255, cv2.THRESH_BINARY
        )
        
        return thresh_mask
    
    def _morphological_processing(self, thresh_mask: np.ndarray) -> np.ndarray:
        """形態學處理，針對桌球大小和形狀優化"""
        # 小目標增強 - 先膨脹再腐蝕
        enhanced = cv2.morphologyEx(
            thresh_mask, cv2.MORPH_DILATE, self.ball_enhance_kernel
        )
        
        # 去除小雜訊
        if self.config.opening_kernel_size_fmo[0] > 0:
            opened = cv2.morphologyEx(
                enhanced, cv2.MORPH_OPEN, self.opening_kernel
            )
        else:
            opened = enhanced
        
        # 連接斷點
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.closing_kernel)
        
        # 針對運動密度進行額外處理
        if self.frame_stats['motion_density'] > 0.1:  # 運動過多時
            # 使用更嚴格的形態學操作
            strict_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, strict_kernel)
        
        return closed
    
    def get_detection_quality(self) -> float:
        """評估當前檢測品質
        
        Returns:
            品質分數 (0-1)，1表示最佳檢測條件
        """
        # 基於多個因素計算品質分數
        brightness_score = 1.0 - abs(self.frame_stats['brightness_mean'] - 128) / 128
        contrast_score = min(1.0, self.frame_stats['contrast_std'] / 50.0)
        motion_score = 1.0 - min(1.0, self.frame_stats['motion_density'] * 5)
        
        return (brightness_score + contrast_score + motion_score) / 3.0
    
    def reset(self) -> None:
        """重置偵測器狀態"""
        self.prev_frames.clear()
        self.threshold_history.clear()
        self.adaptive_threshold = self.config.threshold_value_fmo
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=20, varThreshold=50, detectShadows=False
        )
        self.frame_stats = {
            'brightness_mean': 0,
            'contrast_std': 0,
            'motion_density': 0
        }