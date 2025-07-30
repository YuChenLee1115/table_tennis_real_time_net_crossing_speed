"""
透視校正工具
處理攝像頭透視變形的校正計算
"""

import numpy as np
from typing import Dict
from ..core.config import TrackingConfig


class PerspectiveCorrector:
    """透視校正器"""
    
    def __init__(self, config: TrackingConfig, roi_height: int, roi_bottom_y: int, 
                 roi_start_x: int, roi_end_x: int, frame_width: int):
        self.config = config
        self.roi_height = roi_height
        self.roi_bottom_y = roi_bottom_y
        self.roi_start_x = roi_start_x
        self.roi_end_x = roi_end_x
        self.frame_width = frame_width
        
        # 建立查找表以提高性能
        self.lookup_table = self._create_lookup_table()
        
    def _create_lookup_table(self) -> Dict[int, float]:
        """建立像素到實際距離比例的查找表"""
        lookup_table = {}
        
        for y_in_roi_rounded in range(0, self.roi_height + 1, 10):
            y_global = y_in_roi_rounded + (0 if self.roi_bottom_y == 0 else 0)  # roi_top_y 通常為 0
            ratio = self._calculate_pixel_to_cm_ratio(y_global)
            lookup_table[y_in_roi_rounded] = ratio
            
        return lookup_table
    
    def _calculate_pixel_to_cm_ratio(self, y_global: int) -> float:
        """計算指定Y座標處的像素到公分比例"""
        y_effective = min(y_global, self.roi_bottom_y)
        
        if self.roi_bottom_y == 0:
            relative_y = 0.5
        else:
            relative_y = np.clip(y_effective / self.roi_bottom_y, 0.0, 1.0)
            
        # 根據透視變形計算當前寬度
        current_width_cm = (
            self.config.far_side_width_cm * (1 - relative_y) + 
            self.config.near_side_width_cm * relative_y
        )
        
        # 計算ROI寬度（像素）
        roi_width_px = self.roi_end_x - self.roi_start_x
        
        if current_width_cm > 0 and roi_width_px > 0:
            pixel_to_cm_ratio = current_width_cm / roi_width_px
        else:
            # 回退到標稱比例
            pixel_to_cm_ratio = self.config.table_length_cm / self.frame_width
            
        return pixel_to_cm_ratio
    
    def get_pixel_to_cm_ratio(self, y_global: int) -> float:
        """獲取指定Y座標的像素到公分比例"""
        # 轉換為ROI座標
        y_in_roi = max(0, min(self.roi_height, y_global))
        y_in_roi_rounded = round(y_in_roi / 10) * 10
        
        # 從查找表獲取比例
        if y_in_roi_rounded in self.lookup_table:
            return self.lookup_table[y_in_roi_rounded]
        else:
            # 如果不在查找表中，即時計算
            return self._calculate_pixel_to_cm_ratio(y_global)
    
    def correct_distance(self, pixel_distance: float, y1_global: int, y2_global: int) -> float:
        """校正像素距離為實際距離"""
        ratio1 = self.get_pixel_to_cm_ratio(y1_global)
        ratio2 = self.get_pixel_to_cm_ratio(y2_global)
        avg_ratio = (ratio1 + ratio2) / 2.0
        
        return pixel_distance * avg_ratio