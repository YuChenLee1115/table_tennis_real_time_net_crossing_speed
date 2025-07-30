"""
速度計算器
計算球體的即時速度
"""

import math
from typing import List, Tuple, Optional
from collections import deque
from ..core.config import TrackingConfig
from ..utils.perspective import PerspectiveCorrector


class SpeedCalculator:
    """速度計算器"""
    
    def __init__(self, config: TrackingConfig, perspective_corrector: PerspectiveCorrector):
        self.config = config
        self.perspective_corrector = perspective_corrector
        self.current_speed_kmh = 0.0
        
    def calculate_speed(self, trajectory: List[Tuple[int, int, float]]) -> float:
        """計算當前速度"""
        if len(trajectory) < 2:
            self.current_speed_kmh = 0.0
            return self.current_speed_kmh
            
        # 取最近兩點計算速度
        p1 = trajectory[-2]
        p2 = trajectory[-1]
        
        x1_global, y1_global, t1 = p1
        x2_global, y2_global, t2 = p2
        
        # 計算實際距離（考慮透視校正）
        real_distance_cm = self._calculate_real_distance_cm(
            x1_global, y1_global, x2_global, y2_global
        )
        
        # 計算時間差
        delta_t = t2 - t1
        
        if delta_t > 0.0001:  # 避免除零錯誤
            # 計算速度 (cm/s)
            speed_cm_per_s = real_distance_cm / delta_t
            
            # 轉換為 km/h
            raw_speed_kmh = speed_cm_per_s * self.config.kmh_conversion_factor
            
            # 平滑濾波
            if self.current_speed_kmh > 0:
                self.current_speed_kmh = (
                    (1 - self.config.speed_smoothing_factor) * self.current_speed_kmh +
                    self.config.speed_smoothing_factor * raw_speed_kmh
                )
            else:
                self.current_speed_kmh = raw_speed_kmh
        else:
            # 時間差太小，衰減當前速度
            self.current_speed_kmh *= (1 - self.config.speed_smoothing_factor)
            
        return self.current_speed_kmh
    
    def _calculate_real_distance_cm(self, x1_global: int, y1_global: int, 
                                   x2_global: int, y2_global: int) -> float:
        """計算考慮透視校正的實際距離"""
        # 獲取兩點的像素到實際距離比例
        ratio1 = self.perspective_corrector.get_pixel_to_cm_ratio(y1_global)
        ratio2 = self.perspective_corrector.get_pixel_to_cm_ratio(y2_global)
        avg_ratio = (ratio1 + ratio2) / 2.0
        
        # 計算像素距離
        pixel_distance = math.hypot(x2_global - x1_global, y2_global - y1_global)
        
        # 轉換為實際距離
        real_distance_cm = pixel_distance * avg_ratio
        
        return real_distance_cm
    
    def get_current_speed(self) -> float:
        """獲取當前速度"""
        return self.current_speed_kmh
    
    def reset(self) -> None:
        """重置速度計算器"""
        self.current_speed_kmh = 0.0